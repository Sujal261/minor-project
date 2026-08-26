"""
baseline_mnist.py - the same training with no disty at all.

One process, one machine, the whole global batch in a single forward and
backward. This is the number the distributed run is compared against: same
model, same seed, same data, same batches, same learning rate. The only thing
removed is the ring.

    python baseline_mnist.py --threads 1     one core, like one node gets
    python baseline_mnist.py --threads 0     all cores, PyTorch at full tilt

If the ring is correct, this run and the distributed run must reach the same
accuracy, because they are computing the same gradient every step.
"""

import argparse
import time

import torch

import data_mnist
import train_mnist

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--batch", type=int, default=1000,
                    help="the GLOBAL batch, to match the distributed run")
parser.add_argument("--lr", type=float, default=0.5)
parser.add_argument("--threads", type=int, default=1,
                    help="0 leaves PyTorch's default (all cores)")
args = parser.parse_args()

if args.threads > 0:
    torch.set_num_threads(args.threads)

train_x, train_y = data_mnist.load("train")
test_x, test_y = data_mnist.load("test")
batches = len(train_x) // args.batch

model = train_mnist.build_model()
optimiser = torch.optim.SGD(model.parameters(), lr=args.lr)

start = time.time()
for step in range(args.steps):
    at = (step % batches) * args.batch
    xb = train_x[at:at + args.batch]
    yb = train_y[at:at + args.batch]
    optimiser.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(xb), yb)
    loss.backward()
    optimiser.step()
took = time.time() - start

threads = args.threads if args.threads > 0 else torch.get_num_threads()
print(f"baseline  threads={threads}  global batch={args.batch}  "
      f"steps={args.steps}")
print(f"  {took:.2f}s  ({args.steps / took:.1f} steps/sec, "
      f"{took / args.steps * 1000:.1f} ms/step)")
print(f"  final loss     {float(loss.detach()):.6f}")
print(f"  sum of weights {sum(float(p.detach().sum()) for p in model.parameters()):+.9f}")
print(f"  test accuracy  {train_mnist.accuracy(model, test_x, test_y) * 100:.2f}%")
print(f"  train accuracy {train_mnist.accuracy(model, train_x, train_y) * 100:.2f}%")
