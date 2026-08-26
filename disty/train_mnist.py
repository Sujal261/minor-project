"""
train_mnist.py - the same ring all-reduce as train_ring.py, but training a real
model on real data so the numbers mean something.

    one machine:   python launch_local.py --nodes 3 --script train_mnist.py
    two machines:  python train_mnist.py --rank 0 --peers 10.0.0.1:29600,10.0.0.2:29600
                   python train_mnist.py --rank 1 --peers 10.0.0.1:29600,10.0.0.2:29600

WHAT IS DIFFERENT FROM train_ring.py
Nothing in the middleware. ring.py and allreduce.py are used exactly as they
are, unchanged. The only differences are the model (269,322 numbers instead of
1,377), the data (real MNIST digits instead of a made-up function) and how the
batch is chosen. That is the point: swapping in a real model needed no change to
the transport or the collective.

Two things do become visible at this size that were hidden at 1,377 numbers:

  1. The gradient is 269,322 doubles, which is bigger than allreduce.SLICE
     (262,144), so all_reduce now genuinely loops - two slices per step instead
     of one. The 1,377-number model never exercised that path.

  2. Each block handed to sendall() is about 1 MiB with two nodes, which is far
     larger than a socket's send buffer. This is the size at which the threading
     in Link.exchange stops being a precaution and becomes the only reason the
     program does not deadlock.

HOW THE BATCH IS SPLIT, AND WHY THIS ORDER
Each step picks the GLOBAL batch first and then hands every node a stride of it:

    global batch  = rows [step*G ... step*G + G)        G = batch * world_size
    my rows       = global batch [rank::world_size]

Doing it this way round means the union of every node's rows is exactly the
global batch, with nothing missing and nothing counted twice. So the claim the
check at the bottom tests is precise: the averaged gradient coming out of the
ring equals the gradient one machine would compute over the whole global batch.

Because G is batch*world_size, every node always gets exactly `batch` rows and
there is never a remainder to deal with.
"""

import argparse
import time

import numpy as np
import torch

import allreduce
import data_mnist
import ring


def build_model():
    """
    784 -> 256 -> 256 -> 10, which is 269,322 numbers to average every step.

    784 in because a 28x28 image flattened is 784 pixels, 10 out because there
    are ten digits.

    The seed matters for the same reason as always: every node must start from
    identical weights, or averaging gradients is meaningless.

    .double() puts the model in float64. Real training would use float32 and
    everything here would work the same, but float32 carries only about 7
    digits, so summing half the batch twice and summing the whole batch once
    differ in the 7th digit - which is rounding, not a bug, but it would leave
    the check at the bottom of this file unable to tell the two apart. In
    float64 they agree to about 15 digits and there is nothing to argue about.
    The cost is that every number on the wire is 8 bytes instead of 4.
    """
    torch.manual_seed(1234)
    return torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).double()


def gather_gradients(model):
    """
    Copy every gradient in the model into one flat numpy array.

    The order is whatever model.parameters() gives, and every node builds the
    same model, so every node produces the same order. That is the entire
    protocol - there is no header saying which number is which.

    .numpy() on a float64 CPU tensor is a view, not a copy: it hands back the
    same memory with a numpy label on it. np.concatenate then does one memcpy
    into a fresh array. Compare that with .tolist(), which would have to build
    269,322 separate Python float objects every single step.
    """
    return np.concatenate(
        [parameter.grad.detach().reshape(-1).numpy()
         for parameter in model.parameters()])


def scatter_gradients(model, flat):
    """
    The exact reverse of gather_gradients.

    torch.from_numpy is also a view rather than a copy, so the only real work
    here is grad.copy_(), which is a compiled memcpy. Building a torch tensor
    out of a Python list instead would mean reading every float object back out
    one at a time.
    """
    at = 0
    for parameter in model.parameters():
        count = parameter.grad.numel()
        piece = flat[at:at + count]
        at += count
        parameter.grad.copy_(
            torch.from_numpy(piece).reshape(parameter.grad.shape))
    if at != len(flat):
        raise ValueError(f"had {len(flat)} numbers but the model wanted {at}")


def whole_batch_gradient(inputs, targets):
    """
    The gradient over the whole global batch on one machine, the ordinary
    non-distributed way. This is the answer the ring is supposed to reproduce.
    A fresh model from the same seed, so the weights match what every node
    started this step with.
    """
    model = build_model()
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    return gather_gradients(model)


def accuracy(model, inputs, targets):
    """Fraction of rows the model gets right. No gradients needed."""
    with torch.no_grad():
        predicted = model(inputs).argmax(dim=1)
        return float((predicted == targets).double().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--peers", required=True,
                        help="comma-separated, same on every node")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=500,
                        help="rows per node per step. The global batch is this "
                             "times the number of nodes.")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch compute threads per process. 1 is right "
                             "when several nodes share one machine, because "
                             "otherwise each process grabs every core and they "
                             "fight. Pass 0 on real separate machines to let "
                             "each node use all its cores.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    peers = ring.parse_peers(args.peers)
    world_size = len(peers)
    rank = args.rank

    def say(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    # --- the data ------------------------------------------------------
    train_x, train_y = data_mnist.load("train")
    test_x, test_y = data_mnist.load("test")

    global_batch = args.batch * world_size
    batches = len(train_x) // global_batch
    if batches == 0:
        raise SystemExit(
            f"global batch {global_batch} is bigger than the {len(train_x)} "
            f"rows available - lower --batch")

    say(f"{len(train_x)} training rows, {len(test_x)} test rows")
    say(f"global batch {global_batch} = {args.batch} rows on each of "
        f"{world_size} nodes, {batches} batches per pass")

    # --- the model -----------------------------------------------------
    model = build_model()
    total_numbers = sum(p.numel() for p in model.parameters())
    slices = -(-total_numbers // allreduce.SLICE)
    chunk = -(-min(total_numbers, allreduce.SLICE) // world_size)
    say(f"model has {total_numbers} numbers to average every step")
    say(f"that is {slices} slice(s) of at most {allreduce.SLICE}, so each "
        f"block on the wire is {chunk * allreduce.ITEM} bytes "
        f"({chunk * allreduce.ITEM / 1024:.0f} KiB)")
    say(f"{slices * 2 * (world_size - 1)} exchanges per step, "
        f"{2 * (world_size - 1) / world_size * total_numbers * allreduce.ITEM / 1e6:.2f}"
        f" MB per node per step")

    optimiser = torch.optim.SGD(model.parameters(), lr=args.lr)

    # --- join the ring -------------------------------------------------
    link = ring.connect_ring(rank, peers, verbose=not args.quiet)

    try:
        start_time = time.time()
        for step in range(args.steps):

            # Pick this step's global batch, then take my stride of it. The
            # union of every node's rows is exactly the global batch.
            batch_index = step % batches
            at = batch_index * global_batch
            global_x = train_x[at:at + global_batch]
            global_y = train_y[at:at + global_batch]
            my_x = global_x[rank::world_size]
            my_y = global_y[rank::world_size]

            # 1. forward and backward on MY rows. Ordinary PyTorch - nothing
            #    here knows the ring exists.
            optimiser.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(my_x), my_y)
            loss.backward()

            # 2. flatten, average across the ring, put it back. These three
            #    lines are the entire integration with PyTorch, unchanged from
            #    the toy model.
            mine = gather_gradients(model)
            averaged = allreduce.all_reduce(mine, rank, world_size, link,
                                            average=True)
            scatter_gradients(model, averaged)

            # 3. every node now holds the identical whole-global-batch
            #    gradient, so stepping keeps the model copies identical.
            optimiser.step()

            # The check, on the first step only: does what came out of the ring
            # match the gradient one machine gets from the whole global batch?
            if step == 0:
                wanted = whole_batch_gradient(global_x, global_y)
                worst = float(np.abs(averaged - wanted).max())
                biggest = float(np.abs(wanted).max())
                if worst < 1e-12:
                    say(f"PASS  ring gradient matches the whole-batch "
                        f"gradient over all {global_batch} rows")
                    say(f"      worst difference {worst:.3e}, largest "
                        f"gradient {biggest:.3e}, "
                        f"relative {worst / biggest:.3e}")
                else:
                    say(f"FAIL  ring gradient is wrong - worst difference "
                        f"{worst:.3e}")
                    return 1

            if step % 20 == 0 or step == args.steps - 1:
                weight_sum = sum(float(p.detach().sum())
                                 for p in model.parameters())
                say(f"step {step:4d}  my loss {float(loss.detach()):.6f}   "
                    f"sum of weights {weight_sum:+.9f}")

        took = time.time() - start_time
        say(f"trained {args.steps} steps in {took:.2f}s "
            f"({args.steps / took:.1f} steps/sec, "
            f"{took / args.steps * 1000:.1f} ms/step)")

        # Both of these must be identical on every rank, which is the other
        # half of the correctness story: the copies never drifted apart.
        say(f"test accuracy {accuracy(model, test_x, test_y) * 100:.2f}%  "
            f"on {len(test_x)} unseen digits")
        say(f"train accuracy {accuracy(model, train_x, train_y) * 100:.2f}%")
    finally:
        link.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
