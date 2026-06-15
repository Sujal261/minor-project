"""
worker.py — Type 3: Grad + Weight exchange every SYNC_EVERY batches.

Per-batch behaviour
───────────────────
  Batches 1 to SYNC_EVERY-1:
    forward → backward → optimizer.step()   (local, independent)

  Every SYNC_EVERY batches (and at the last batch of each epoch):
    STEP A — send weights to master, receive averaged weights, reset model
    STEP B — send last-batch grads to master, receive averaged grads,
              apply one extra optimizer step from the shared weight position

Timing logged
─────────────
  Per non-sync batch : compute_time, local_step_time, batch_total
  Per sync batch     : compute_time, weight_send_wait, grad_send_wait,
                       apply_time, sync_batch_total
  Per epoch          : epoch_duration, local_compute, sync_wait, apply
  Overall            : grand_total, communication_overhead %
"""

import os
import socket
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from comm import recv_object, send_object
from model import MnistCNN

# ── Config ──────────────────────────────────────────────────────────────────────
WORKER_ID   = int(os.environ["WORKER_ID"])
MASTER_HOST = os.environ.get("MASTER_HOST", "master")
MASTER_PORT = int(os.environ.get("MASTER_PORT", 29500))
NUM_EPOCHS  = int(os.environ.get("NUM_EPOCHS", 3))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", 64))
LR          = float(os.environ.get("LR", 0.01))
MOMENTUM    = float(os.environ.get("MOMENTUM", 0.9))
SYNC_EVERY  = int(os.environ.get("SYNC_EVERY", 5))
MNIST_ROOT  = os.environ.get("MNIST_ROOT", "/tmp/mnist")
SEED        = 42 + WORKER_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

def LOG(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][worker-{WORKER_ID}] {msg}", flush=True)


# ── Data ─────────────────────────────────────────────────────────────────────────
def build_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full    = datasets.MNIST(MNIST_ROOT, train=True, download=True, transform=transform)
    n       = len(full)
    half    = n // 2
    indices = range(0, half) if WORKER_ID == 0 else range(half, n)
    return DataLoader(Subset(full, list(indices)), batch_size=BATCH_SIZE, shuffle=True)


def build_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(MNIST_ROOT, train=False, download=True, transform=transform)
    return DataLoader(test_ds, batch_size=256, shuffle=False)


# ── Evaluation ───────────────────────────────────────────────────────────────────
def evaluate(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += yb.size(0)
    return correct / total * 100


def weight_drift(model_a, model_b):
    return sum(
        (p1.data - p2.data).norm().item()
        for p1, p2 in zip(model_a.parameters(), model_b.parameters())
    )


# ── Connect to master ────────────────────────────────────────────────────────────
def connect_to_master(retries: int = 30, delay: float = 2.0) -> socket.socket:
    for attempt in range(1, retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((MASTER_HOST, MASTER_PORT))
            LOG(f"Connected to master at {MASTER_HOST}:{MASTER_PORT}")
            return sock
        except ConnectionRefusedError:
            LOG(f"Master not ready (attempt {attempt}/{retries}), retrying in {delay}s …")
            time.sleep(delay)
    raise RuntimeError("Could not connect to master after all retries.")


# ── Training loop ────────────────────────────────────────────────────────────────
def train(sock: socket.socket):
    loader      = build_loader()
    test_loader = build_test_loader()

    model     = MnistCNN().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    LOG(f"Shard   : {len(loader.dataset)} samples | {len(loader)} batches/epoch | device={DEVICE}")
    LOG(f"Config  : LR={LR}  momentum={MOMENTUM}  batch_size={BATCH_SIZE}  "
        f"epochs={NUM_EPOCHS}  sync_every={SYNC_EVERY}")

    # ── Overall timing accumulators ───────────────────────────────────────────
    total_compute_time      = 0.0   # forward + backward across all batches
    total_local_step_time   = 0.0   # optimizer.step() on non-sync batches
    total_sync_wait_time    = 0.0   # total time blocked on master (weight + grad)
    total_apply_time        = 0.0   # final averaged-grad step on sync batches
    total_syncs             = 0

    grand_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()

        total_loss      = correct = total = 0
        n_batches       = 0
        n_syncs_epoch   = 0
        ep_compute      = 0.0
        ep_local_step   = 0.0
        ep_sync_wait    = 0.0
        ep_apply        = 0.0

        # Keep last batch's grads so we can send them at sync time
        last_grads = None

        LOG(f"{'='*60}")
        LOG(f"Epoch {epoch}/{NUM_EPOCHS} starting  |  "
            f"sync every {SYNC_EVERY} batches")
        LOG(f"{'='*60}")

        for batch_idx, (xb, yb) in enumerate(loader):
            batch_1indexed  = batch_idx + 1          # 1-based for display
            is_sync = (batch_1indexed % SYNC_EVERY == 0) or (batch_1indexed == len(loader))
            batch_wall_start = time.time()

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            # ── 1. Forward + backward (ALL batches) ───────────────────────
            t0 = time.time()
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            compute_time = time.time() - t0

            # Capture this batch's grads (needed at sync time)
            last_grads = {
                name: param.grad.detach().clone()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

            # ── 2a. NON-SYNC batch: local optimizer step, no master comms ─
            if not is_sync:
                t1 = time.time()
                optimizer.step()
                local_step_time  = time.time() - t1
                batch_total      = time.time() - batch_wall_start

                LOG(f"epoch={epoch} batch={batch_1indexed}/{len(loader)} [local]  "
                    f"compute={compute_time*1000:.1f}ms  "
                    f"local_step={local_step_time*1000:.1f}ms  "
                    f"batch_total={batch_total*1000:.1f}ms")

                total_compute_time    += compute_time
                total_local_step_time += local_step_time
                ep_compute            += compute_time
                ep_local_step         += local_step_time

            # ── 2b. SYNC batch: weight exchange + grad exchange ────────────
            else:
                LOG(f"epoch={epoch} batch={batch_1indexed}/{len(loader)} "
                    f"[SYNC #{n_syncs_epoch+1}] starting weight + grad exchange")

                # ── STEP A: send weights, receive averaged weights ─────────
                current_weights = {
                    name: param.data.detach().clone()
                    for name, param in model.named_parameters()
                }

                t_wa = time.time()
                LOG(f"  STEP-A sending weights to master...")
                send_object(sock, {
                    "type":  "weights",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "data":  current_weights,
                })
                resp_w       = recv_object(sock)
                weight_wait  = time.time() - t_wa
                avg_weights  = resp_w["averaged_weights"]

                # Reset model to averaged weights — drift → 0
                for name, param in model.named_parameters():
                    param.data.copy_(avg_weights[name].to(DEVICE))

                LOG(f"  STEP-A received averaged weights  "
                    f"(weight_send_wait={weight_wait*1000:.1f}ms)")

                # ── STEP B: send last-batch grads, receive averaged grads ──
                t_gb = time.time()
                LOG(f"  STEP-B sending grads to master...")
                send_object(sock, {
                    "type":  "grads",
                    "epoch": epoch,
                    "batch": batch_idx,
                    "data":  last_grads,
                })
                resp_g      = recv_object(sock)
                grad_wait   = time.time() - t_gb
                avg_grads   = resp_g["averaged_grads"]

                LOG(f"  STEP-B received averaged grads  "
                    f"(grad_send_wait={grad_wait*1000:.1f}ms)")

                # ── Apply averaged grads from the shared weight position ───
                t_apply = time.time()
                optimizer.zero_grad()
                for name, param in model.named_parameters():
                    if name in avg_grads:
                        param.grad = avg_grads[name].to(DEVICE)
                optimizer.step()
                apply_time = time.time() - t_apply

                sync_wait   = weight_wait + grad_wait
                batch_total = time.time() - batch_wall_start

                LOG(f"epoch={epoch} batch={batch_1indexed}/{len(loader)} [SYNC] DONE | "
                    f"compute={compute_time*1000:.1f}ms  "
                    f"weight_wait={weight_wait*1000:.1f}ms  "
                    f"grad_wait={grad_wait*1000:.1f}ms  "
                    f"apply={apply_time*1000:.1f}ms  "
                    f"sync_total={batch_total*1000:.1f}ms")

                total_compute_time   += compute_time
                total_sync_wait_time += sync_wait
                total_apply_time     += apply_time
                ep_compute           += compute_time
                ep_sync_wait         += sync_wait
                ep_apply             += apply_time
                n_syncs_epoch        += 1
                total_syncs          += 1

            total_loss += loss.item()
            correct    += (out.argmax(1) == yb).sum().item()
            total      += yb.size(0)
            n_batches  += 1

        # ── End of epoch ──────────────────────────────────────────────────
        eval_start = time.time()
        test_acc   = evaluate(model, test_loader)
        eval_time  = time.time() - eval_start
        epoch_dur  = time.time() - epoch_start

        LOG(f"{'─'*60}")
        LOG(f"Epoch {epoch}/{NUM_EPOCHS} COMPLETE  |  {n_syncs_epoch} syncs this epoch")
        LOG(f"  train_loss     = {total_loss/n_batches:.4f}")
        LOG(f"  train_acc      = {correct/total*100:.2f}%")
        LOG(f"  test_acc       = {test_acc:.2f}%")
        LOG(f"  epoch_duration = {epoch_dur:.2f}s")
        LOG(f"  ├─ compute     = {ep_compute:.2f}s  ({ep_compute/epoch_dur*100:.1f}%)")
        LOG(f"  ├─ local_steps = {ep_local_step:.2f}s  ({ep_local_step/epoch_dur*100:.1f}%)")
        LOG(f"  ├─ sync_wait   = {ep_sync_wait:.2f}s  ({ep_sync_wait/epoch_dur*100:.1f}%)")
        LOG(f"  ├─ apply       = {ep_apply:.2f}s  ({ep_apply/epoch_dur*100:.1f}%)")
        LOG(f"  └─ eval        = {eval_time:.2f}s")
        LOG(f"{'─'*60}")

    # ── Final summary ─────────────────────────────────────────────────────────
    grand_total = time.time() - grand_start

    LOG(f"{'='*60}")
    LOG(f"TRAINING COMPLETE")
    LOG(f"  grand_total_time       = {grand_total:.2f}s")
    LOG(f"  total_compute          = {total_compute_time:.2f}s  "
        f"({total_compute_time/grand_total*100:.1f}%)")
    LOG(f"  total_local_steps      = {total_local_step_time:.2f}s  "
        f"({total_local_step_time/grand_total*100:.1f}%)")
    LOG(f"  total_sync_wait        = {total_sync_wait_time:.2f}s  "
        f"({total_sync_wait_time/grand_total*100:.1f}%)")
    LOG(f"  total_apply            = {total_apply_time:.2f}s  "
        f"({total_apply_time/grand_total*100:.1f}%)")
    LOG(f"  total_syncs            = {total_syncs}")
    LOG(f"  communication_overhead = {total_sync_wait_time/grand_total*100:.1f}%  "
        f"(only paid {total_syncs}x vs every batch before)")
    LOG(f"{'='*60}")

    send_object(sock, "DONE")
    LOG("Sent DONE to master. Exiting.")


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sock = connect_to_master()
    try:
        train(sock)
    finally:
        sock.close()