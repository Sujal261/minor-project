"""
trace_ring.py - one complete ring all-reduce, printed number by number.

    python launch_local.py --nodes 3 --script trace_ring.py

This does exactly what train_ring.py does - real PyTorch, real backward pass,
real sockets - but with a model so small that its whole gradient is six numbers,
so you can watch every one of them move round the ring.

WHY THE NUMBERS COME OUT WHOLE
Real gradients are ugly decimals, which makes a trace impossible to follow. So
this picks a setup where the maths happens to be exact, without faking anything:

  the model    torch.nn.Linear(2, 2), weights and bias set to zero
  the input    x = [1, 2], the same on every node
  the target   different on every node, so every node gets a DIFFERENT gradient
               (which is the whole point - otherwise there is nothing to average)

With the weights at zero the prediction is zero, so for mean-squared-error loss
the gradient works out by hand as:

    d(loss)/d(weight[j][k]) = -target[j] * x[k]
    d(loss)/d(bias[j])      = -target[j]

Node r uses target = [-(2r+1), -(2r+2)], so with three nodes:

    rank 0  target [-1 -2]  ->  gradient [1  2  2  4  1  2]
    rank 1  target [-3 -4]  ->  gradient [3  6  4  8  3  4]
    rank 2  target [-5 -6]  ->  gradient [5 10  6 12  5  6]
                                         ---------------
                              total      [9 18 12 24  9 12]
                              average    [3  6  4  8  3  4]

That last line is what all three nodes must end up with. The backward pass
really does compute those gradients - nothing is hard-coded - and the script
checks the answer at the end.

WHERE THE SIX NUMBERS COME FROM
A Linear(2, 2) has two parameters of different shapes:

    weight  2x2 matrix        bias  2 numbers
    [[w00 w01]                      [b0 b1]
     [w10 w11]]

gather_gradients flattens them, in the order model.parameters() gives, into one
run of six:

    [ w00  w01  w10  w11 | b0  b1 ]
      0    1    2    3     4   5      <- position in the flat list

and the ring cuts that into one chunk per node - with three nodes, two each:

    chunk 0        chunk 1        chunk 2
    [w00 w01]      [w10 w11]      [b0 b1]

The chunks have nothing to do with the layers. Chunk 2 happens to be the bias
here, but that is a coincidence of the sizes - the ring sees a flat run of
numbers and cuts it into equal pieces, and neither it nor the model cares where
the boundaries land.
"""

import argparse

import torch

import allreduce
import ring

# The same two functions train_ring.py uses - imported rather than copied, so
# this really is tracing the code that does the work.
from train_ring import gather_gradients, scatter_gradients


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--peers", required=True)
    parser.add_argument("--quiet", action="store_true",
                        help="ignored - this script always traces")
    args = parser.parse_args()

    peers = ring.parse_peers(args.peers)
    world_size = len(peers)
    rank = args.rank

    def say(msg=""):
        print(f"[rank {rank}] {msg}", flush=True)

    def heading(msg):
        say()
        say(f"===== {msg} =====")

    # The model. Zeroed so the arithmetic is exact, and identical on every node -
    # which it has to be, or averaging gradients would be meaningless.
    model = torch.nn.Linear(2, 2).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()

    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    # My slice of the data. A different target on every node is what makes every
    # node compute a different gradient.
    inputs = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    targets = torch.tensor([[-(2.0 * rank + 1), -(2.0 * rank + 2)]],
                           dtype=torch.float64)

    heading("who I am and what data I have")
    say(f"rank {rank} of {world_size}")
    say(f"input  x = {inputs.tolist()[0]}   (same on every node)")
    say(f"target t = {targets.tolist()[0]}   (DIFFERENT on every node)")

    # Join the ring. Nothing is sent yet - this only opens the two sockets.
    link = ring.connect_ring(rank, peers, verbose=False)
    say(f"ring ready: I receive from rank {link.prev_rank}, "
        f"I send to rank {link.next_rank}")

    try:
        # --- stock PyTorch. Not one line of this knows the ring exists. ----
        heading("1. ordinary PyTorch: forward, loss, backward")
        optimiser.zero_grad()
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        say(f"prediction = {predictions.tolist()[0]}  "
            f"(zero, because the weights are zero)")
        say(f"my loss    = {float(loss.detach()):g}")
        say("gradients now sitting in the model, one per parameter:")
        for name, parameter in model.named_parameters():
            say(f"  {name:<7} shape {tuple(parameter.shape)}  "
                f"grad {parameter.grad.tolist()}")

        heading("2. flatten every gradient into one list")
        mine = gather_gradients(model)
        say(f"flat gradient ({len(mine)} numbers): {[f'{v:g}' for v in mine]}")
        say("this is MY gradient, from MY slice of the data - not the one we "
            "want to step with")

        heading("3. the ring all-reduce (watch the chunks)")
        averaged = allreduce.all_reduce(mine, rank, world_size, link,
                                        average=True, verbose=True)

        heading("4. what came back")
        say(f"averaged gradient: {[f'{v:g}' for v in averaged]}")

        # The hand-worked answer, for any number of nodes. Averaging
        # target = [-(2r+1), -(2r+2)] over r = 0..N-1 gives [-N, -(N+1)], and
        # the gradient is -target[j] * x[k] for the weights and -target[j] for
        # the bias, with x = [1, 2]. For three nodes this is [3 6 4 8 3 4].
        n = world_size
        wanted = [n, 2 * n, n + 1, 2 * (n + 1), n, n + 1]
        if all(abs(a - w) < 1e-12 for a, w in zip(averaged, wanted)):
            say(f"PASS  matches the hand-worked average {wanted}")
        else:
            say(f"FAIL  expected {wanted}")
            return 1

        heading("5. put the averaged numbers back into the model")
        scatter_gradients(model, averaged)
        for name, parameter in model.named_parameters():
            say(f"  {name:<7} grad now {parameter.grad.tolist()}")

        # --- stock PyTorch again ------------------------------------------
        heading("6. ordinary PyTorch: step")
        optimiser.step()
        for name, parameter in model.named_parameters():
            say(f"  {name:<7} = {parameter.detach().tolist()}")
        weight_sum = sum(float(p.detach().sum()) for p in model.parameters())
        say(f"sum of all weights = {weight_sum:+.12f}")
        say("every node must print that same number - identical weights, "
            "identical update, no drift")
    finally:
        link.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
