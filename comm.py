"""
comm.py — low-level send/recv helpers for gradient exchange.

Wire format (per message):
  [8 bytes] total payload length (big-endian uint64)
  [N bytes] pickle-serialised Python object

Both master and workers import this module.
"""

import io
import pickle
import socket
import struct


# ── primitives ────────────────────────────────────────────────────────────────

def send_object(sock: socket.socket, obj) -> None:
    """Serialise obj with pickle and send it over sock."""
    payload = pickle.dumps(obj)
    header  = struct.pack(">Q", len(payload))   # 8-byte big-endian length
    sock.sendall(header + payload)


def recv_object(sock: socket.socket):
    """Receive one object from sock (blocking)."""
    header = _recv_exactly(sock, 8)
    length = struct.unpack(">Q", header)[0]
    payload = _recv_exactly(sock, length)
    return pickle.loads(payload)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from sock, blocking until available."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed before all bytes arrived.")
        buf.extend(chunk)
    return bytes(buf)
