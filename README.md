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

The ring code itself cannot tell which one it got — `exchange()` is the same
either way. `--tcp` forces `127.0.0.1` if you want to see that path on Linux.

The detection is one line in `ring.py`:

```python
def local_sockets_work():
    return hasattr(socket, "AF_UNIX")
```

Python only defines `socket.AF_UNIX` on platforms that have it, so checking for
the name is more reliable than checking the OS name.

One WSL2 caveat, since that is a Linux that lives inside Windows: in `mirrored`
networking mode it routes `127.0.0.1` through Windows rather than real loopback

```bash
ip route get 127.0.0.1
# -> 127.0.0.1 via 169.254.73.152 dev loopback0
```

and that path drops packets once a few processes push tens of megabytes at each
other. So on WSL2, use the default (local sockets) rather than `--tcp`. Real
Windows and real Linux are both unaffected.

## Speed, honestly

Single-process training is faster than this ring, and you should expect that:

| | steps/sec |
|---|---|
| One process | ~2400 |
| 3 nodes, one thread each | ~500 |
| 3 nodes, PyTorch's default threads | ~16 |

The model is 1,377 numbers and one forward+backward is 0.42 ms, while the four
socket exchanges per step cost about 1.6 ms. Communication dominates completely,
and the three processes share the same cores so no extra compute is won. This
demo shows the mechanism working correctly, not a speedup. All-reduce pays off
when per-step compute vastly exceeds per-step communication — a large model, on
devices with their own cores.

That last row is why `train_ring.py` calls `torch.set_num_threads(1)`: without
it the processes fight over the same cores and lose 30× for nothing.

## What are `.gitignore` and `__pycache__`?

Nothing you wrote and nothing you need. When Python imports a file it saves a
pre-chewed copy next to it in `__pycache__/` so the next import is faster. It is
generated automatically, it is per-Python-version, and it is rebuilt whenever
you change the source — so committing it is pointless and it can even confuse
someone on a different Python. `.gitignore` is a plain list of names for git to
ignore, and it is there precisely so `__pycache__` never gets pushed. You can
delete `__pycache__` any time; Python will just make it again.

## Verified

- Correctness suite clean at 2, 3, 4, 5, 6, 7, 8 and 11 nodes.
- Training gradient matches the single-machine whole-batch gradient to
  ~4e-16 at every node count tested.
- 45 MB across 3 nodes: 0.48 s, 60 MB per node each way — exactly the
  `2×(N−1)/N` the formula predicts. 10 consecutive runs, no failures.

Run on **real Windows** (Python 3.12.10, torch 2.13.0+cpu,
`sys.platform == 'win32'`), not just Linux:

- `hasattr(socket, "AF_UNIX")` is `False` there, so it took the TCP branch by
  itself and reported `talking over 127.0.0.1 (TCP)`. No flags given.
- Correctness suite clean at 2, 3, 5 and 7 nodes — 20/55/77 PASS, zero FAIL.
- 45 MB across 3 nodes over TCP: clean, 60 MB per node each way, 44 MB/s. This
  is the exact case WSL2 could not survive, which is the whole point — real
  Windows has real loopback.
- `train_ring.py` gradient check PASS (1.11e-16) and `trace_ring.py` matches the
  hand-worked `[3, 6, 4, 8, 3, 4]`.

**A genuinely mixed ring works.** Ranks 0 and 1 on Windows Python, rank 2 on
Linux Python, same `127.0.0.1` ring: 30 PASS, zero FAIL. The wire format is
byte-identical across the two — both little-endian, both 8-byte doubles:

```
Linux  : byteorder little  itemsize 8  1.5,-2.25 -> 000000000000f83f00000000000002c0
Windows: byteorder little  itemsize 8  1.5,-2.25 -> 000000000000f83f00000000000002c0
```

## One real caveat for mixed devices

`build_model()` relies on `torch.manual_seed(1234)` giving every node identical
starting weights. That holds when every node runs the **same PyTorch version**,
and it is not guaranteed when they don't. Measured, same seed, same code:

```
torch 2.12.1 (Linux)  : -0.069368191063404083
torch 2.13.0 (Windows): -0.069368183612823486
```

Those are adjacent float32 values, 1 ULP apart — the versions compute `Linear`'s
initialiser slightly differently. Averaging gradients keeps nodes in step only if
they *started* in step, so mismatched initial weights mean the nodes are training
subtly different models from step 0.

For one machine, or several machines with identical PyTorch, this never bites. To
be safe on genuinely mixed devices, don't trust the seed — have rank 0 send its
weights round the ring once before training and have everyone adopt them.
`all_reduce` already does exactly the right thing if you reduce rank 0's weights
with everyone else's zeroed, or just reduce all the weights and divide by N.

Two things deliberately **not** claimed:

- The TCP path has never been run between two physically separate machines. Same
  code, but a real network is not the same as loopback.
- macOS has not been tested at all. It is POSIX with `AF_UNIX` and
  little-endian, so it takes the same branch Linux does, but that is reasoning
  rather than evidence.
