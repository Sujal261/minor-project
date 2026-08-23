"""
train_ring.py - train a real PyTorch model across several processes, using the
ring all-reduce in this folder to average the gradients.

    python launch_local.py --nodes 3 --script train_ring.py

WHAT DATA PARALLEL TRAINING IS
Every node holds an identical copy of the model, and the data is split up so
each node sees a different slice. Each node runs an ordinary forward and
backward pass on its own slice, which gives it a gradient - but one computed
from only part of the batch, so it is not the gradient we want.

The gradient we want is the one we would have got from the whole batch at once.
Because the loss is a mean, that is exactly the average of the per-node
gradients. So: average them across the ring, put the average back, and step.
Every node applies the identical update to identical weights, so the copies stay
in lockstep for ever and nobody ever sends a weight anywhere.

That averaging is the all-reduce, and it is the only communication in the whole
of distributed training. One call per step.

WHAT WE DO WITH THE GRADIENT
A model's gradient is not one array, it is one array per layer, all different
shapes. The ring does not care about any of that - it wants a flat run of
numbers. So every step:

    1. copy every gradient into one flat list          gather_gradients
    2. all-reduce that list, asking for the average    allreduce.all_reduce
    3. copy the averaged numbers back into the model   scatter_gradients
    4. optimiser.step()

Steps 1 and 3 are the "serialising", and note how dull it is. There is no
protocol and no format, because both ends run this same code over lists of the
same length in the same order. That is the only reason it works, and it is why
every node must build the same model the same way.

HOW WE KNOW IT IS RIGHT
  - On the first step each node also works out the gradient over the FULL
    dataset by itself and compares it with what came out of the ring. If the
    ring is right those must match to floating-point noise. This is the real
    test: it says the distributed gradient IS the single-machine gradient.
  - Every step each node prints the sum of all its weights. Every node must
    print the same number, so any drift between the copies shows up at once.
"""

import argparse
import time

import torch

import allreduce
import ring

# One compute thread per process, and this line is worth more than it looks.
#
# PyTorch defaults to about one thread per CPU core. That is right for one
# process on its own, but here we start several processes on ONE computer, so
# each of them grabs every core and they spend their time fighting each other
# instead of working. Measured on a 20-core machine, 3 nodes: 16 steps/sec with
# the default, 500 steps/sec with this line. Thirty times faster for one line.
#
# It has to be here rather than an environment variable because
# "OMP_NUM_THREADS=1 python ..." is not a valid command on Windows.
#
# On real separate devices you would delete this - there each node has its own
# cores and should use all of them.
torch.set_num_threads(1)


def build_model():
    """
    8 -> 32 -> 32 -> 1, which is 1,377 numbers to average every step. Small on
    purpose: this file is about the communication, not the model.

    The seed matters. PyTorch initialises layers randomly, so without it the
    nodes would start from different models and averaging gradients would be
    meaningless.

    .double() puts the model in float64. Real training uses float32 and this
    would work the same, but float32 only carries about 7 digits, so summing a
    third of the batch three times and summing the whole batch at once differ in
    the 7th digit. That is rounding, not a bug - but it would leave the check at
    the bottom of this file unable to tell the two apart. In float64 the answers
    agree to about 15 digits and there is nothing to argue about.
    """
    torch.manual_seed(1234)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).double()


def make_data(rows=600, features=8):
    """
    The full dataset, identical on every node because of the seed. In real life
    each node would read its own data off its own disk and nobody would ever
    hold all of it - this is only so the check below has something to compare
    against.
    """
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn(rows, features, generator=generator,
                         dtype=torch.float64)
    # Some arbitrary function for the model to learn.
    targets = (inputs[:, :1] * 2.0
               - inputs[:, 1:2]
               + 0.5 * inputs[:, 2:3]).tanh()
    return inputs, targets


def gather_gradients(model):
    """
    Copy every gradient in the model into one flat list of plain floats.

    reshape(-1) flattens a tensor of any shape into a single row, and tolist()
    turns it into ordinary Python floats. Slow for a huge model, but completely
    transparent - there is no encoding step, and you can print the list and read
    it.

    The order is whatever model.parameters() gives, which is the order the layers
    were built in. Every node builds the same model, so every node produces the
    same order. That is the entire "protocol".
    """
    flat = []
    for parameter in model.parameters():
        flat.extend(parameter.grad.reshape(-1).tolist())
    return flat


def scatter_gradients(model, flat):
    """
    The exact reverse: walk the parameters in the same order, take as many
    numbers off the front of the list as each one needs, and write them back into
    the gradient in the right shape.
    """
    at = 0
    for parameter in model.parameters():
        count = parameter.grad.numel()
        piece = flat[at:at + count]
        at += count
        parameter.grad.copy_(
            torch.tensor(piece, dtype=parameter.grad.dtype)
            .reshape(parameter.grad.shape))
    if at != len(flat):
        raise ValueError(f"had {len(flat)} numbers but the model wanted {at}")


def full_dataset_gradient(inputs, targets):
    """
    The gradient over the whole dataset on one machine, the ordinary
    non-distributed way. Only used for the check on the first step - this is the
    answer the ring is supposed to be reproducing.
    """
    model = build_model()
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    return gather_gradients(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--peers", required=True,
                        help="comma-separated, same on every node")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    peers = ring.parse_peers(args.peers)
    world_size = len(peers)
    rank = args.rank

    def say(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    # --- the data ------------------------------------------------------
    inputs, targets = make_data()

    # Every node must get the SAME number of rows. Averaging gradients only
    # equals the whole-batch gradient if the batches are the same size, so if
    # the dataset does not divide evenly, drop the leftover rows rather than
    # give some nodes an extra one.
    usable = len(inputs) - (len(inputs) % world_size)
    inputs = inputs[:usable]
    targets = targets[:usable]

    # rank 0 takes rows 0, 3, 6...; rank 1 takes 1, 4, 7...; and so on.
    my_inputs = inputs[rank::world_size]
    my_targets = targets[rank::world_size]
    say(f"{len(inputs)} rows in use, {len(my_inputs)} of them mine")

    # --- the model -----------------------------------------------------
    model = build_model()
    total_numbers = sum(p.numel() for p in model.parameters())
    say(f"model has {total_numbers} numbers to average every step")

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    # --- join the ring -------------------------------------------------
    link = ring.connect_ring(rank, peers, verbose=not args.quiet)

    try:
        start = time.time()
        for step in range(args.steps):

            # 1. forward and backward on MY slice. Completely ordinary PyTorch -
            #    nothing here knows the ring exists.
            optimiser.zero_grad()
            predictions = model(my_inputs)
            loss = torch.nn.functional.mse_loss(predictions, my_targets)
            loss.backward()

            # 2. the gradient we now hold came from part of the batch. Flatten
            #    it, average it across the ring, put it back. These three lines
            #    are the entire integration with PyTorch.
            mine = gather_gradients(model)
            averaged = allreduce.all_reduce(mine, rank, world_size, link,
                                            average=True)
            scatter_gradients(model, averaged)

            # 3. every node now holds the identical whole-batch gradient, so
            #    stepping keeps the copies of the model identical.
            optimiser.step()

            # detach() because these are still attached to the autograd graph
            # and we only want the number.
            weight_sum = sum(float(p.detach().sum())
                             for p in model.parameters())
            say(f"step {step:2d}  my loss {float(loss.detach()):.6f}   "
                f"sum of weights {weight_sum:+.12f}")

            # The check, on the first step only: does what came out of the ring
            # match the gradient you would get from the whole dataset at once?
            if step == 0:
                wanted = full_dataset_gradient(inputs, targets)
                worst = max(abs(a - b) for a, b in zip(averaged, wanted))
                if worst < 1e-12:
                    say(f"PASS  ring gradient matches the whole-batch "
                        f"gradient (worst difference {worst:.2e})")
                else:
                    say(f"FAIL  ring gradient is wrong - worst difference "
                        f"{worst:.2e}")
                    return 1

        took = time.time() - start
        say(f"trained {args.steps} steps in {took:.2f}s "
            f"({args.steps / took:.1f} steps/sec)")
        say("all done - the 'sum of weights' lines should be identical across "
            "every rank on every step")
    finally:
        # Always tidy up, even if something above went wrong, so a failed run
        # does not leave the next one waiting on a socket nobody is holding.
        link.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
