# disty — ring all-reduce

Distributed gradient averaging, written from scratch in plain Python so you can
read every part of it. No `torch.distributed`, no NCCL, no MPI — just sockets,
`array('d')`, and two loops.

## What you need

Python 3 and PyTorch. Nothing else — no `torch.distributed`, no MPI, no NCCL.

```bash
pip install torch
```

`run.py` and `launch_local.py` don't even need PyTorch; only `train_ring.py` and
`trace_ring.py` do.

## Run it

Three processes on one machine, each a real node with real sockets:

```bash
python launch_local.py --nodes 3
```

Train a real PyTorch model across three nodes:

```bash
python launch_local.py --nodes 3 --script train_ring.py
```

Watch one all-reduce happen number by number:

```bash
python launch_local.py --nodes 3 --script trace_ring.py
```

Any node count works — `--nodes 2`, `--nodes 7`, `--nodes 11`.

These exact commands work on Windows, Linux and macOS — see below.

## On real devices

Give every device the **same** `--peers` list and change only `--rank`:

```bash
python run.py --rank 0 --peers 10.0.0.1:29600,10.0.0.2:29600
```

Position in the list *is* the rank. So the machine above is rank 0, and the one
you launch with `--rank 1` is rank 1.

## The files

| file | what it does |
|---|---|
| `ring.py` | Networking only. Connects the nodes into a ring and swaps blocks of bytes with the two neighbours. Never does arithmetic. |
| `allreduce.py` | The algorithm only. Reduce-scatter lap, then all-gather lap. Never touches a socket. |
| `train_ring.py` | A real PyTorch model trained across the ring. This is the point of the other two. |
| `trace_ring.py` | The same thing with a six-number gradient, printing every value at every step. Read this first. |
| `run.py` | Correctness tests, ending with a 45 MB transfer. |
| `launch_local.py` | Starts N processes on this machine so you can try it without N machines. |

## How it works, in short

Every node computes a gradient from its own slice of the data, so every node has
a *different* gradient. The one we want is the average. Getting there:

1. **Cut** the flat gradient into N equal chunks, one per node.
2. **Lap 1 — reduce-scatter** (N−1 steps): pass a chunk on, *add* the chunk that
   arrives. Each chunk collects another node's numbers every time it moves. At
   the end each node holds exactly one finished chunk, a different one each.
3. **Lap 2 — all-gather** (N−1 steps): pass the finished chunks round,
   *overwriting* instead of adding, until everyone has all N.
4. **Divide** by N.

Every node moves `2 × (N−1)/N × gradient_size` — under 2× its own gradient no
matter how many nodes there are, and no link busier than any other. That is the
whole reason for the ring instead of one node doing the averaging.

## The integration with PyTorch is three lines

Nothing in the training loop is special. In `train_ring.py`:

```python
optimiser.zero_grad()                                     # stock
loss = torch.nn.functional.mse_loss(model(x), y)          # stock
loss.backward()                                           # stock

mine = gather_gradients(model)                                    # inserted
averaged = allreduce.all_reduce(mine, rank, world_size, link)     # inserted
scatter_gradients(model, averaged)                                # inserted

optimiser.step()                                          # stock
```

No autograd hooks, no model wrapper. PyTorch leaves a gradient in `p.grad`, we
replace it with a better one, PyTorch steps. It never learns anything happened.

## Two things worth knowing

**Padding.** If the gradient length doesn't divide by N, `allreduce.py` rounds it
up with zeros first (at most N−1 of them) and cuts them off at the end. That
makes every chunk identical in size, so the block you send always matches the
block you receive — which is why there are no length headers anywhere in the
protocol.

**The send is on a thread.** `sendall()` doesn't return until the OS has room for
all the bytes, and it only gets room as the other node reads. If every node sent
first and read second, every node would be stuck in `sendall()` waiting for a
neighbour who is also stuck in `sendall()`. Sending on a thread means each node
sends and receives at the same time. See the comment in `ring.py`.

## Runs on Windows, Linux and macOS

No flags, no changes, no configuration. `launch_local.py` asks Python what this
computer supports and picks accordingly:

| | how the nodes talk | why |
|---|---|---|
| Linux, macOS | local sockets in the temp folder | never touches the network stack, so it is faster and immune to network trouble |
| Windows | `127.0.0.1` | Windows has no local sockets; its loopback is real, so TCP is fine |



## Verified

- Correctness suite clean at 2, 3, 4, 5, 6, 7, 8 and 11 nodes.
- Training gradient matches the single-machine whole-batch gradient to
  ~4e-16 at every node count tested.
- 45 MB across 3 nodes: 0.48 s, 60 MB per node each way — exactly the
  `2×(N−1)/N` the formula predicts. 10 consecutive runs, no failures.

