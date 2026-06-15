"""
master.py — Type 3: handles weight exchange + grad exchange at sync points.

Each sync round has TWO sub-rounds per (epoch, batch):
  Sub-round "weights" : collect weights from both workers → average → send back
  Sub-round "grads"   : collect grads  from both workers → average → send back

Workers always send weights first, wait for reply, then send grads.
The master handles them as two sequential RoundBuffers per sync point.

Timing logged
─────────────
  Per sync:
    weight_skew  : gap between first and second worker sending weights
    weight_avg   : time to compute averaged weights
    grad_skew    : gap between first and second worker sending grads
    grad_avg     : time to compute averaged grads
  Overall:
    master uptime, total syncs handled
"""

import os
import socket
import threading
import time

from comm import recv_object, send_object

# ── Config ───────────────────────────────────────────────────────────────────────
HOST        = "0.0.0.0"
PORT        = int(os.environ.get("MASTER_PORT", 29500))
NUM_WORKERS = 2
NUM_EPOCHS  = int(os.environ.get("NUM_EPOCHS", 3))

def LOG(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][master] {msg}", flush=True)


# ── Generic RoundBuffer (reused for both weights and grads) ──────────────────────
class RoundBuffer:
    """
    Collects tensors from all workers then fires the average.
    Works for both weight dicts and grad dicts — same logic.
    """

    def __init__(self, num_workers: int, label: str):
        self.num_workers  = num_workers
        self.label        = label           # "weights" or "grads" — for logging
        self._lock        = threading.Lock()
        self._data        = {}              # worker_id -> tensor dict
        self._arrive_time = {}
        self._ready       = threading.Event()
        self._avg         = None
        self._first_arrive = None

    def submit(self, worker_id: int, tensor_dict: dict):
        now = time.time()
        with self._lock:
            self._data[worker_id]        = tensor_dict
            self._arrive_time[worker_id] = now

            if len(self._data) == 1:
                self._first_arrive = now
                LOG(f"[{self.label}] worker {worker_id} arrived FIRST")

            elif len(self._data) == self.num_workers:
                skew = now - self._first_arrive
                LOG(f"[{self.label}] worker {worker_id} arrived LAST  "
                    f"skew={skew*1000:.1f}ms")

                t0           = time.time()
                self._avg    = self._average()
                avg_time     = time.time() - t0

                LOG(f"[{self.label}] averaging done in {avg_time*1000:.2f}ms  "
                    f"unblocking workers")
                self._ready.set()

    def wait_for_average(self) -> dict:
        self._ready.wait()
        return self._avg

    def _average(self) -> dict:
        keys = list(next(iter(self._data.values())).keys())
        avg  = {}
        for k in keys:
            tensors = [self._data[wid][k] for wid in sorted(self._data)]
            avg[k]  = sum(tensors) / len(tensors)
        return avg


# ── SyncSequencer: manages weight + grad buffers per (epoch, batch) ─────────────
class SyncSequencer:
    """
    Each sync point is keyed by (epoch, batch).
    Two RoundBuffers per key: one for weights, one for grads.
    Cleans up after all workers have received both replies.
    """

    def __init__(self, num_workers: int):
        self.num_workers  = num_workers
        self._lock        = threading.Lock()
        self._weight_bufs = {}    # (epoch, batch) -> RoundBuffer
        self._grad_bufs   = {}    # (epoch, batch) -> RoundBuffer
        self._sent_w      = {}    # (epoch, batch) -> int
        self._sent_g      = {}    # (epoch, batch) -> int

    def _ensure_key(self, key):
        if key not in self._weight_bufs:
            self._weight_bufs[key] = RoundBuffer(self.num_workers, "weights")
            self._grad_bufs[key]   = RoundBuffer(self.num_workers, "grads")
            self._sent_w[key]      = 0
            self._sent_g[key]      = 0

    def get_weight_buf(self, epoch: int, batch: int) -> RoundBuffer:
        key = (epoch, batch)
        with self._lock:
            self._ensure_key(key)
        return self._weight_bufs[key]

    def get_grad_buf(self, epoch: int, batch: int) -> RoundBuffer:
        key = (epoch, batch)
        with self._lock:
            self._ensure_key(key)
        return self._grad_bufs[key]

    def mark_weight_sent(self, epoch: int, batch: int):
        key = (epoch, batch)
        with self._lock:
            self._sent_w[key] += 1
            self._try_cleanup(key)

    def mark_grad_sent(self, epoch: int, batch: int):
        key = (epoch, batch)
        with self._lock:
            self._sent_g[key] += 1
            self._try_cleanup(key)

    def _try_cleanup(self, key):
        # Only clean up once all workers received BOTH weight and grad replies
        if (self._sent_w.get(key, 0) == self.num_workers and
                self._sent_g.get(key, 0) == self.num_workers):
            del self._weight_bufs[key]
            del self._grad_bufs[key]
            del self._sent_w[key]
            del self._sent_g[key]
            LOG(f"Cleaned up sync buffers for key={key}")


# ── Worker handler thread ─────────────────────────────────────────────────────────
def handle_worker(conn: socket.socket, addr, worker_id: int,
                  sequencer: SyncSequencer, master_start: float):
    LOG(f"Worker {worker_id} connected from {addr}")
    total_syncs = 0

    try:
        while True:
            t_recv = time.time()
            msg    = recv_object(conn)
            recv_time = time.time() - t_recv

            if msg == "DONE":
                uptime = time.time() - master_start
                LOG(f"Worker {worker_id} sent DONE  |  "
                    f"syncs handled={total_syncs}  master_uptime={uptime:.2f}s")
                break

            msg_type = msg["type"]    # "weights" or "grads"
            epoch    = msg["epoch"]
            batch    = msg["batch"]
            data     = msg["data"]

            LOG(f"Received [{msg_type}] from worker {worker_id}  "
                f"epoch={epoch} batch={batch+1}  recv_time={recv_time*1000:.1f}ms")

            if msg_type == "weights":
                # ── Weight sub-round ───────────────────────────────────────
                buf = sequencer.get_weight_buf(epoch, batch)
                buf.submit(worker_id, data)
                avg_weights = buf.wait_for_average()

                t_send = time.time()
                send_object(conn, {
                    "averaged_weights": avg_weights,
                    "epoch": epoch, "batch": batch,
                })
                send_time = time.time() - t_send

                LOG(f"Sent averaged weights to worker {worker_id}  "
                    f"epoch={epoch} batch={batch+1}  send_time={send_time*1000:.1f}ms")
                sequencer.mark_weight_sent(epoch, batch)

            elif msg_type == "grads":
                # ── Grad sub-round ─────────────────────────────────────────
                buf = sequencer.get_grad_buf(epoch, batch)
                buf.submit(worker_id, data)
                avg_grads = buf.wait_for_average()

                t_send = time.time()
                send_object(conn, {
                    "averaged_grads": avg_grads,
                    "epoch": epoch, "batch": batch,
                })
                send_time = time.time() - t_send

                LOG(f"Sent averaged grads to worker {worker_id}  "
                    f"epoch={epoch} batch={batch+1}  send_time={send_time*1000:.1f}ms")
                sequencer.mark_grad_sent(epoch, batch)
                total_syncs += 1

    except (ConnectionError, EOFError) as e:
        LOG(f"Worker {worker_id} connection error: {e}")
    finally:
        conn.close()
        LOG(f"Connection closed for worker {worker_id}.")


# ── Main ─────────────────────────────────────────────────────────────────────────
def main():
    master_start = time.time()
    LOG(f"Starting on {HOST}:{PORT}  "
        f"(expecting {NUM_WORKERS} workers, {NUM_EPOCHS} epochs)")
    LOG(f"Protocol: weight exchange + grad exchange at each sync point")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(NUM_WORKERS)

    sequencer = SyncSequencer(NUM_WORKERS)

    threads = []
    for worker_id in range(NUM_WORKERS):
        LOG(f"Waiting for worker {worker_id} to connect …")
        conn, addr = server.accept()
        t = threading.Thread(
            target=handle_worker,
            args=(conn, addr, worker_id, sequencer, master_start),
            daemon=True,
        )
        t.start()
        threads.append(t)

    LOG(f"All {NUM_WORKERS} workers connected. Training in progress …")

    for t in threads:
        t.join()

    total_uptime = time.time() - master_start
    LOG(f"{'='*60}")
    LOG(f"All workers finished.")
    LOG(f"Total master uptime = {total_uptime:.2f}s")
    LOG(f"{'='*60}")

    server.close()


if __name__ == "__main__":
    main()