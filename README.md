# Distributed AllReduce Training (Master + 2 Workers)

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Docker Network                        │
│                                                              │
│   ┌─────────┐   grads ──→   ┌──────────┐   ←── grads       │
│   │worker-0 │               │  master  │                    │
│   │shard 0  │   ←── avg ──  │ averager │  ── avg ──→        │
│   └─────────┘               └──────────┘          ┌───────┐ │
│                                                    │worker-1│ │
│                                                    │shard 1│ │
│                                                    └───────┘ │
└──────────────────────────────────────────────────────────────┘
```

Per-batch protocol:
1. Each worker does forward + backward → raw gradients
2. Both workers send raw gradients to master (TCP socket)
3. Master waits until BOTH grads arrive, computes element-wise average
4. Master sends averaged gradients back to each worker
5. Workers overwrite `.grad` buffers with averaged grads → `optimizer.step()`

Workers are blocked at step 4 — they cannot proceed until master
finishes averaging. This guarantees weights never diverge.

## Files

```
.
├── comm.py            # shared send/recv helpers (pickle over TCP)
├── model.py           # MnistCNN definition (shared by workers)
├── master.py          # parameter server
├── worker.py          # training worker
├── Dockerfile.master  # image for master
├── Dockerfile.worker  # image for both workers
└── docker-compose.yml # wires everything together
```

## Quick Start

```bash
docker compose up --build
```

Logs are interleaved. To follow a single container:

```bash
docker compose logs -f worker-0
docker compose logs -f master
```

## Override Hyperparameters

All hyperparameters can be set as environment variables without rebuilding:

```bash
NUM_EPOCHS=5 LR=0.005 BATCH_SIZE=128 docker compose up --build
```

| Variable    | Default | Description              |
|-------------|---------|--------------------------|
| NUM_EPOCHS  | 3       | Training epochs          |
| BATCH_SIZE  | 64      | Mini-batch size per shard|
| LR          | 0.01    | SGD learning rate        |
| MOMENTUM    | 0.9     | SGD momentum             |

## How the Synchronisation Works

```
worker-0                master                  worker-1
   |                      |                        |
   |── grads (batch 0) ──>|                        |
   |   [BLOCKED]          |<── grads (batch 0) ────|
   |                      |   [BLOCKED]            |
   |                      | avg = (g0 + g1) / 2    |
   |<── avg_grads ────────|──── avg_grads ─────>   |
   | optimizer.step()     |                optimizer.step()
   |                      |                        |
   |── grads (batch 1) ──>|                        |
   ...
```

The master uses a `threading.Event` inside `RoundBuffer` — each worker
thread blocks on `wait_for_average()` until the second worker's
gradients land, then both are unblocked simultaneously.

## Extending

- **More workers**: increase `NUM_WORKERS` in `docker-compose.yml` and
  add more `worker-N` services. The master's `RoundBuffer` automatically
  waits for all `NUM_WORKERS` gradients before averaging.
- **Sync every N batches**: accumulate gradients locally for N steps,
  send the accumulated sum, reset. Adjust `worker.py` accordingly.
- **GPU support**: change the base image to `pytorch/pytorch:latest` and
  add `deploy.resources.reservations.devices` in `docker-compose.yml`.
