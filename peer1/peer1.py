import argparse
import socket
import struct
import threading
import time

import torch
from torch import nn

BUFFER = 4096
PEER_INFO = b'\x00'
START = b'\xcc'
PEER_ALIVE = b'\xbb'
EPOCHS = 12
LOCAL_SAMPLES = 256
BATCH_SIZE = 32
INPUT_SIZE = 64
HIDDEN_SIZE = 128
DEFAULT_PEER_IP = "127.0.0.1"
DEFAULT_LPORT = 37000
DEFAULT_FPORT = 20000


def receive_exact(conn, size):
    data = b""
    while len(data) < size:
        part = conn.recv(min(size - len(data), BUFFER))
        if not part:
            raise ConnectionError("connection closed during receive")
        data += part
    return data


def create_model():
    torch.manual_seed(2026)
    return nn.Sequential(nn.Linear(INPUT_SIZE, HIDDEN_SIZE), nn.ReLU(), nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(), nn.Linear(HIDDEN_SIZE, 2))


def create_data(rank):
    generator = torch.Generator().manual_seed(7000 + rank)
    features = torch.randn(LOCAL_SAMPLES, INPUT_SIZE, generator=generator)
    labels = ((features[:, :8].sum(1) + features[:, 8:16].sum(1)) > 0).long()
    return features, labels


def send_values(conn, values):
    conn.sendall(struct.pack(f"!{len(values)}f", *values))


def receive_values(conn, count):
    return list(struct.unpack(f"!{count}f", receive_exact(conn, count * 4)))


def ring_allreduce(values, peer_ip, lport, rank, ring_size, next_addr):
    original_size = len(values)
    chunk_size = (original_size + ring_size - 1) // ring_size
    padded = values + [0.0] * (chunk_size * ring_size - original_size)
    chunks = [padded[i * chunk_size:(i + 1) * chunk_size] for i in range(ring_size)]
    incoming = {}

    def accept_previous():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((peer_ip, lport))
            listener.listen(1)
            incoming["connection"], _ = listener.accept()

    accept_thread = threading.Thread(target=accept_previous)
    accept_thread.start()
    outgoing = None
    while outgoing is None:
        try:
            candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            candidate.connect(next_addr)
            outgoing = candidate
        except OSError:
            time.sleep(0.1)
    accept_thread.join()
    incoming_connection = incoming["connection"]

    for step in range(ring_size - 1):
        send_index = (rank - step) % ring_size
        receive_index = (rank - step - 1) % ring_size
        send_values(outgoing, chunks[send_index])
        received = receive_values(incoming_connection, chunk_size)
        chunks[receive_index] = [a + b for a, b in zip(chunks[receive_index], received)]

    for step in range(ring_size - 1):
        send_index = (rank - step + 1) % ring_size
        receive_index = (rank - step) % ring_size
        send_values(outgoing, chunks[send_index])
        chunks[receive_index] = receive_values(incoming_connection, chunk_size)

    outgoing.close()
    incoming_connection.close()
    return [value / ring_size for chunk in chunks for value in chunk][:original_size]


def flatten_gradients(model):
    return [value for parameter in model.parameters() for value in parameter.grad.detach().float().reshape(-1).tolist()]


def restore_gradients(model, values):
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.grad = torch.tensor(values[offset:offset + count], dtype=parameter.dtype).reshape(parameter.shape)
        offset += count


def train(model_args):
    torch.set_num_threads(1)
    model = create_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    loss_function = nn.CrossEntropyLoss()
    features, labels = create_data(model_args.rank)
    print(f"[rank {model_args.rank}] training on {model_args.peer_ip}; parameters={sum(p.numel() for p in model.parameters())}")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for start in range(0, LOCAL_SAMPLES, BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features[start:start + BATCH_SIZE]), labels[start:start + BATCH_SIZE])
            loss.backward()
            averaged = ring_allreduce(flatten_gradients(model), model_args.peer_ip, model_args.lport, model_args.rank, model_args.ring_size, model_args.next_addr)
            restore_gradients(model, averaged)
            optimizer.step()
            total_loss += loss.item()
        if epoch == 0 or (epoch + 1) % 3 == 0 or epoch == EPOCHS - 1:
            print(f"[rank {model_args.rank}] epoch={epoch + 1:02d} loss={total_loss / (LOCAL_SAMPLES / BATCH_SIZE):.4f}")

    with torch.no_grad():
        accuracy = (model(features).argmax(1) == labels).float().mean().item()
        checksum = sum(parameter.float().sum().item() for parameter in model.parameters())
    print(f"[rank {model_args.rank}] final_accuracy={accuracy:.3f} model_checksum={checksum:.6f}")


def control_listener(args, training_lock):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.peer_ip, args.fport))
        listener.listen()
        while True:
            connection, address = listener.accept()
            try:
                if address[0] != args.server_ip or receive_exact(connection, 1) != START:
                    continue
                ring_size, rank = struct.unpack("!HH", receive_exact(connection, 4))
                next_ip, next_port, _previous_ip, _previous_port = struct.unpack("!4sH4sH", receive_exact(connection, 12))
                next_addr = (socket.inet_ntoa(next_ip), next_port)
                print(f"[rank {rank}] START ring_size={ring_size} next={next_addr}")
                if ring_size >= 2 and training_lock.acquire(blocking=False):
                    run_args = argparse.Namespace(**vars(args), rank=rank, ring_size=ring_size, next_addr=next_addr)
                    threading.Thread(target=training_round, args=(training_lock, run_args), daemon=True).start()
            finally:
                connection.close()


def training_round(lock, args):
    try:
        train(args)
    finally:
        lock.release()


def heartbeat(args):
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
                connection.bind((args.peer_ip, 0))
                connection.connect((args.server_ip, args.server_port))
                connection.sendall(PEER_ALIVE)
        except OSError:
            pass
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9999)
    parser.add_argument("--peer-ip", default=DEFAULT_PEER_IP)
    parser.add_argument("--lport", type=int, default=DEFAULT_LPORT)
    parser.add_argument("--fport", type=int, default=DEFAULT_FPORT)
    args = parser.parse_args()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.bind((args.peer_ip, 0))
        connection.connect((args.server_ip, args.server_port))
        message = struct.pack("!HH", args.lport, args.fport)
        connection.sendall(struct.pack("!cI", PEER_INFO, len(message)) + message)
    lock = threading.Lock()
    threading.Thread(target=control_listener, args=(args, lock), daemon=True).start()
    threading.Thread(target=heartbeat, args=(args,), daemon=True).start()
    print(f"peer registered peer_ip={args.peer_ip} lport={args.lport} fport={args.fport}")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
