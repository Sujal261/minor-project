"""
ring.py - the networking. Two jobs: connect the nodes into a ring, and swap a
block of bytes with your two neighbours. The maths is in allreduce.py and never
touches a socket.

    rank 0  ->  rank 1  ->  rank 2
      ^                       |
      +-----------------------+

So every node keeps two sockets open:

    send_sock  ->  the next node     (rank + 1)
    recv_sock  <-  the previous node (rank - 1)

That is the whole topology. No master, no server doing the averaging - every
node is the same kind of thing.
"""

import os
import socket
import threading
import time


class Link:
    """
    One node's two sockets.

    Only one method matters from outside: exchange(data) sends `data` to the
    next node and returns the same number of bytes from the previous node.
    """

    def __init__(self, send_sock, recv_sock, rank, next_rank, prev_rank):
        self.send_sock = send_sock      # -> next node
        self.recv_sock = recv_sock      # <- previous node
        self.rank = rank
        self.next_rank = next_rank
        self.prev_rank = prev_rank

    def exchange(self, data):
        """
        Send `data` to the next node, and return the same number of bytes from
        the previous node.

        WHY THE SENDING IS ON A THREAD
        This is the one genuinely tricky thing in the file, and it is not
        optional. sendall() does not return until the operating system has
        somewhere to put all the bytes, and the OS only makes room as the node
        on the other end reads them. A block here is around 700 KB and the
        buffer for a connection is a few hundred KB, so sendall() cannot finish
        on its own.

        Now imagine every node sent first and read second. Every node would be
        sat inside sendall(), waiting for a neighbour to read - and that
        neighbour is also sat inside sendall(), waiting for its neighbour. The
        whole ring locks up and nothing ever moves again.

        Putting the send on a thread means we are sending and receiving at the
        same time, so the bytes we owe our neighbour keep flowing while we are
        collecting the bytes we are owed.

        Both blocks are always the same size, which is why there is no length
        header anywhere - allreduce.py pads everything so the chunks come out
        equal.
        """
        # A one-item list because a thread cannot "return" anything - this is
        # just somewhere for it to leave an error if the send fails.
        trouble = []

        def send():
            try:
                self.send_sock.sendall(data)
            except OSError as problem:
                trouble.append(problem)

        sender = threading.Thread(target=send)
        sender.start()

        incoming = self.receive_exactly(len(data))

        sender.join()
        if trouble:
            raise trouble[0]
        return incoming

    def receive_exactly(self, count):
        """
        Read exactly `count` bytes.

        recv() hands over whatever has arrived so far, which is usually less
        than you asked for, so keep going until the lot is in.
        """
        bag = bytearray()
        while len(bag) < count:
            piece = self.recv_sock.recv(count - len(bag))
            if not piece:
                # Empty bytes means the other end closed the connection.
                raise ConnectionError(
                    f"rank {self.rank}: rank {self.prev_rank} hung up with "
                    f"{count - len(bag)} bytes still owed")
            bag.extend(piece)
        return bytes(bag)

    def close(self):
        """
        Shut down without breaking a neighbour that is still working.

        Plain close() is not enough. If a socket still has bytes in it that
        nobody read, the operating system gives up on the connection and throws
        away anything it had not delivered yet - so a slower neighbour sees the
        connection vanish in the middle of a block. Doing it politely instead:
        say "nothing more from me", then read whatever was still in flight.
        """
        try:
            self.send_sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass

        # Drain anything the previous node had already sent, until it hangs up
        # too. The timeout is so a node that died rather than finished does not
        # leave us waiting for a goodbye that is never coming.
        try:
            self.recv_sock.settimeout(5.0)
            while self.recv_sock.recv(65536):
                pass
        except OSError:
            pass

        self.send_sock.close()
        self.recv_sock.close()


def connect_ring(rank, peers, verbose=True):
    """
    Join the ring. Returns a Link.

    rank  - which node am I: 0, 1, 2, ...
    peers - one address per rank, THE SAME LIST on every node. peers[rank] is
            the address I listen on; the rest I dial.

    Each node has to listen (so the previous node can reach it) and dial onwards
    (to the next node), and those have to overlap. If a node waited to be
    connected to before connecting onwards, every node would be waiting and
    nobody would ever dial. So the listening goes on a thread while this thread
    dials.
    """
    world_size = len(peers)
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size

    def log(msg):
        if verbose:
            print(f"[rank {rank}] {msg}", flush=True)

    # Listen FIRST, so if the previous node arrives early the operating system
    # holds the connection for us instead of refusing it.
    listener = new_socket(peers[rank])
    if is_tcp(peers[rank]):
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", peers[rank][1]))
    else:
        # A local socket is a file, so a leftover one from a crashed run would
        # block the bind. Nothing else could be using this name.
        if os.path.exists(peers[rank]):
            os.unlink(peers[rank])
        listener.bind(peers[rank])
    listener.listen(8)
    log(f"listening on {peers[rank]}, waiting for rank {prev_rank}")

    # Accept on a thread, dial on this one.
    accepted = {}

    def accept_previous():
        conn, address = listener.accept()
        accepted["sock"] = conn

    accepter = threading.Thread(target=accept_previous)
    accepter.start()

    send_sock = dial(peers[next_rank], next_rank, log)

    accepter.join()
    recv_sock = accepted["sock"]
    log(f"rank {prev_rank} connected")

    if is_tcp(peers[rank]):
        no_delay(send_sock)
        no_delay(recv_sock)

    listener.close()       # the ring is built, we do not need this any more
    log(f"ring ready: recv from rank {prev_rank}, send to rank {next_rank}")
    return Link(send_sock, recv_sock, rank, next_rank, prev_rank)


def is_tcp(address):
    """A ("host", port) pair means TCP; a plain string means a local socket."""
    return isinstance(address, tuple)


def local_sockets_work():
    """
    Does this computer have local sockets?

    Linux and macOS do. Windows mostly does not: Python only defines
    socket.AF_UNIX on platforms that support it, so asking whether the name
    exists is the reliable way to find out - you never have to know which OS
    you are on. launch_local.py calls this to pick a default that works
    everywhere.
    """
    return hasattr(socket, "AF_UNIX")


def new_socket(address):
    """
    An unconnected socket of the right kind for this address.

    Two kinds, and the rest of this file cannot tell them apart:

    AF_INET is ordinary TCP over the network. This is what you want between
    real machines, and it works on every operating system.

    AF_UNIX is a local socket - same send and recv, but both ends must be on
    one computer and it never touches the network stack at all, so it is faster
    and nothing the network does can disturb it. Linux and macOS only.

    On this machine (WSL2 in "mirrored" networking mode) the local socket is not
    just faster, it is necessary. WSL2 routes 127.0.0.1 through Windows instead
    of using real loopback:

        ip route get 127.0.0.1
        -> 127.0.0.1 via 169.254.73.152 dev loopback0

    and that path starts dropping packets once a few processes push tens of
    megabytes at each other, showing up as a stall or "connection reset by peer"
    at a random point. It is not fixable from Python. Real Windows and real
    Linux both have real loopback, so 127.0.0.1 is fine there.
    """
    if is_tcp(address):
        return socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if not local_sockets_work():
        raise RuntimeError(
            f"asked for a local socket ({address}) but this computer does not "
            f"have them - most likely Windows. Use addresses with a port "
            f"instead, like 127.0.0.1:29600, or run launch_local.py which "
            f"picks for you.")
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


def no_delay(sock):
    """
    Send immediately instead of waiting to bundle small messages together.

    By default TCP holds a small send back for a moment in case more is coming,
    so it can pack it into one packet. Sensible for typing in a terminal,
    useless here: the node on the other side is waiting for this exact block
    and will not send anything until it arrives, so there is nothing to bundle
    with. Both sides just wait, and every message picks up about 40
    milliseconds of delay for nothing.
    """
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def dial(address, target_rank, log):
    """
    Connect to the next node, retrying until it is up.

    Nodes never start at the same instant, so "connection refused" (or, for a
    local socket, "no such file") just means "not launched yet". A socket that
    failed to connect cannot be reused, hence the close() before trying again.
    """
    while True:
        sock = new_socket(address)
        try:
            sock.connect(address)
            log(f"connected to rank {target_rank} at {address}")
            return sock
        except OSError:
            sock.close()
            log(f"rank {target_rank} at {address} not up yet, retrying...")
            time.sleep(0.5)


def parse_peers(text):
    """
    Turn one comma-separated string into a list of addresses, one per rank.
    Position in the list IS the rank, so every node must be given the same
    string in the same order.

    Two forms, told apart by whether there is a port on the end:

      "10.0.0.1:29600,10.0.0.2:29600"        -> real machines, over TCP
      "/tmp/disty-0.sock,/tmp/disty-1.sock"  -> one machine, local sockets
    """
    peers = []
    for piece in text.split(","):
        piece = piece.strip()
        host, _, port = piece.rpartition(":")
        if host and port.isdigit():
            peers.append((host, int(port)))
        else:
            peers.append(piece)
    return peers
