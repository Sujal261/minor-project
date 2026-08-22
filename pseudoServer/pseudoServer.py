import argparse
import time
import struct
import socket
import threading
import os
import sys
from dataclasses import dataclass, field

BUFFER = 4096

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=9999)
parser.add_argument("--auto-start", type=float)
args = parser.parse_args()

HOST = args.host
PORT = args.port

PEER_INFO = b'\x00'
PEER_INQUIRY = b'\xff'
PEER_ALIVE = b'\xbb'
START = b'\xcc'


def recv_exact(conn, n_bytes):
    data = b""
    while len(data) < n_bytes:
        frag = conn.recv(n_bytes - len(data))
        if not frag:
            raise ConnectionError("socket closed while reading")
        data += frag
    return data


@dataclass
class Worker:
    ip_addr: str
    listening_port: int
    fault_port: int
    last_update_time: float = field(default_factory=time.time)


peer_list = []
peer_list_lock = threading.Lock()


def peer_alive_check():
    curr_time = time.time()
    with peer_list_lock:
        survivors = [p for p in peer_list if curr_time - p.last_update_time < 30.0]
        changed = len(peer_list) != len(survivors)
        if changed:
            peer_list[:] = survivors

    if changed:
        ring_formation()

def peer_alive_check_thread_fn():
    while True:
        peer_alive_check()
        time.sleep(10)


def ring_formation():
    with peer_list_lock:
        peers = list(peer_list)

    peer_list_len = len(peers)
    if peer_list_len < 2:
        print(f"need at least 2 peers to form a ring, currently have {peer_list_len}")
        return

    for i in range(peer_list_len):
        current_peer = peers[i]
        sending_peer = peers[(i + 1) % peer_list_len]
        receiving_peer = peers[(i - 1 + peer_list_len) % peer_list_len]

        sending_ip_bytes = socket.inet_aton(sending_peer.ip_addr)
        receiving_ip_bytes = socket.inet_aton(receiving_peer.ip_addr)

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ring_socket:
                ring_socket.connect((current_peer.ip_addr, current_peer.fault_port))

                message = struct.pack(
                    '!cHH4sH4sH',
                    START,
                    peer_list_len,
                    i,
                    sending_ip_bytes,
                    sending_peer.listening_port,
                    receiving_ip_bytes,
                    receiving_peer.listening_port
                )

                ring_socket.sendall(message)
        except OSError as exc:
            print(f"failed START to {current_peer.ip_addr}:{current_peer.fault_port}: {exc}")

    print(f"ring formed with {peer_list_len} peers, START dispatched to all")

def exit_thread_fn():
    while True:
        command = input("$ ")
        if "exit" in command:
            os._exit(0)


def find_peer_by_ip(peer_addr_tuple):
    for peer in peer_list:
        if peer.ip_addr == peer_addr_tuple[0]:
            return peer
    return None


def peer_thread_function(peer_conn, peer_addr):
    with peer_conn:
        packet_status = recv_exact(peer_conn, 1)

        if packet_status == PEER_ALIVE:
            new_time = time.time()
            with peer_list_lock:
                curr_peer = find_peer_by_ip(peer_addr)
                if curr_peer is not None:
                    curr_peer.last_update_time = new_time

        if packet_status == PEER_INFO:
            message_size, = struct.unpack('!I', recv_exact(peer_conn, 4))
            listening_port, fault_port = struct.unpack('!HH', recv_exact(peer_conn, message_size))

            updated = False
            with peer_list_lock:
                for existing in peer_list:
                    if existing.ip_addr == peer_addr[0] and existing.fault_port == fault_port:
                        existing.listening_port = listening_port
                        existing.last_update_time = time.time()
                        updated = True
                        break

                if not updated:
                    peer_info = Worker(
                        ip_addr=peer_addr[0],
                        listening_port=listening_port,
                        fault_port=fault_port,
                        last_update_time=time.time()
                    )
                    peer_list.append(peer_info)
                    print(f"registered peer {peer_info}")

def start_func():
    while True:
        command = input("$ ")
        if "start" in command:
            ring_formation()
        if "exit" in command:
            os._exit(0)


def auto_start_fn(delay):
    time.sleep(delay)
    print(f"[auto-start] triggering ring_formation with {len(peer_list)} peers")
    ring_formation()


if args.auto_start is not None:
    threading.Thread(target=auto_start_fn, args=(args.auto_start,), daemon=True).start()
else:
    start_thread = threading.Thread(target=start_func, daemon=True)
    start_thread.start()

peer_alive_check_thread = threading.Thread(target=peer_alive_check_thread_fn, daemon=True)
peer_alive_check_thread.start()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"pseudoServer listening on {HOST}:{PORT}")

    while True:
        peer_conn, peer_addr = server_socket.accept()
        peer_thread = threading.Thread(target=peer_thread_function, args=(peer_conn, peer_addr))
        peer_thread.start()

