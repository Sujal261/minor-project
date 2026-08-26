"""
allreduce.py - the ring all-reduce itself. No sockets in here at all.

WHAT IT DOES
Every node starts with its own run of numbers, all the same length. When this
finishes, every node holds the element-wise total (or average) of everybody's
numbers. In training those numbers are the gradients: each node worked out a
gradient from its own slice of the data, and averaging them is what makes the
nodes behave like one big batch instead of drifting into different models.

WHY A RING AND NOT A MASTER
The easy version is: everyone sends their gradient to one node, it adds them up,
it sends the answer back. That works, but the master has to receive N gradients
and send N back, so it gets slower the more nodes you add - the opposite of why
you added them.

A ring has no master. Every node only ever talks to its two neighbours, and
every node moves the same amount of data:

    2 * (N-1)/N * (size of the gradient)

Look at (N-1)/N - it never reaches 1. Three nodes or a thousand, no node ever
moves more than twice its own gradient, and no link is busier than any other.

HOW IT WORKS
Cut the numbers into N equal chunks, one per node, then do two laps.

  Lap 1, "reduce-scatter" (N-1 steps): pass a chunk on, ADD the chunk that
  arrives into yours. Each chunk collects another node's numbers every time it
  moves. After N-1 steps each node holds exactly one finished chunk - and a
  different one from everybody else.

  Lap 2, "all-gather" (N-1 steps): pass the finished chunks round, this time
  OVERWRITING instead of adding, since they need no more work. After N-1 steps
  everyone has all N.

WHY A NUMPY ARRAY AND NOT A PYTHON LIST
A numpy float64 array holds raw 8-byte doubles laid end to end in the machine's
own byte order, so handing one to a socket is just .tobytes() - no encoding
step, nothing hidden, and nothing to agree with the other side about. A list of
the same numbers would be a list of pointers to millions of separate float
objects, and turning that into bytes would have to copy and rebuild every one of
them. float64 is used throughout because a Python float already IS a C double,
so nothing is rounded on the way in or out.

The other reason is speed, and it only shows up at a realistic size. The two
places this file does arithmetic - adding the chunk that arrives, and dividing
at the end - are one line each with numpy:

    numbers[at:at + chunk] += theirs          instead of a Python for loop
    result /= world_size                      instead of a Python for loop

Written as Python loops those two lines cost about 40 ms per step on a
269,322-number gradient, which is more than the network does. Written as numpy
they cost almost nothing, because numpy runs the same additions in compiled
code. Both are element-wise operations on the same doubles in the same order, so
the answers are not merely close - they are bit for bit what the loops gave.
"""

import numpy as np


ITEM = 8            # bytes per number, because a float64 is an 8-byte double

# How many numbers to reduce in one go. 262144 doubles is 2 MB, so with three
# nodes each block on the wire is about 700 KB. Doing a whole 45 MB gradient in
# one go would mean handing 45 MB to a socket, and the operating system's buffer
# for a connection is only a few hundred KB, so it would have to be dribbled out
# over hundreds of writes while the other node waited. Slices cost nothing - the
# same bytes cross the same wires either way.
SLICE = 1 << 18

# With verbose=True, print the actual numbers only for slices this short.
# Above it you would get millions of numbers on one line.
SHOW_LIMIT = 24


def pad_length(total, world_size):
    """
    Round `total` up to the next multiple of world_size.

    Cutting 10 numbers across 3 nodes would give chunks of 4, 3, 3 - and then
    the block you send is a different size from the block you get back, so both
    sides need length headers and the code fills up with bookkeeping. Padding
    with a few zeros instead makes every chunk identical, so there is nothing to
    agree about. Zeros are safe: adding one changes nothing, and they get cut
    off at the end.
    """
    chunks = (total + world_size - 1) // world_size      # round up
    return chunks * world_size


def all_reduce(values, rank, world_size, link, average=True, verbose=False):
    """
    Add up `values` across every node in the ring. Every node must pass the same
    number of values. Returns a numpy float64 array of the totals, or of the
    averages if average=True.

    `values` may be a numpy array, a Python list, or an array('d') - anything
    numpy can read. The result is always a fresh writable float64 array, so the
    caller can hand it straight to torch.from_numpy without another copy.

    `link` is the Link from ring.connect_ring - our two neighbours.

    All this does is walk the numbers in slices and hand each one to
    reduce_one_slice below. Every node slices identically, so every node is
    always working on the same slice as everybody else.
    """
    # asarray alone would share memory with a numpy caller, and we are about to
    # write into this, so take our own copy. astype always copies.
    numbers = np.asarray(values, dtype=np.float64).astype(np.float64, copy=True)

    # One node is not a ring - nobody to talk to and nothing to add.
    if world_size == 1:
        return numbers

    pieces = []
    for start in range(0, len(numbers), SLICE):
        part = numbers[start:start + SLICE]
        if verbose:
            print(f"[rank {rank}] slice {start}..{start + len(part)}",
                  flush=True)
        pieces.append(reduce_one_slice(part, rank, world_size, link, verbose))

    # One concatenate at the end instead of growing an array slice by slice.
    result = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    # Turn the totals into averages. Done once at the end rather than before
    # sending, so there is one division pass instead of one per step.
    if average:
        result /= world_size

    return result


def reduce_one_slice(numbers, rank, world_size, link, verbose=False):
    """
    The ring all-reduce on one slice. `numbers` is a float64 array; returns a
    new float64 array of the totals across all nodes.
    """
    total = len(numbers)

    # Pad with zeros so the slice divides into N equal chunks. Allocating the
    # padded array and copying into the front of it is the numpy way of saying
    # "numbers + [0.0] * padding", and it also gives us our own copy, so the
    # caller's array is left alone.
    padded = pad_length(total, world_size)
    work = np.zeros(padded, dtype=np.float64)
    work[:total] = numbers
    numbers = work
    chunk = padded // world_size

    # --- printing helpers, only used when verbose ------------------------
    def say(msg):
        if verbose:
            print(f"[rank {rank}] {msg}", flush=True)

    def tracing():
        return verbose and total <= SHOW_LIMIT

    def fmt(values):
        return "[" + " ".join(f"{v:g}" for v in values) + "]"

    def show(label):
        if tracing():
            pieces = []
            for i in range(world_size):
                pieces.append(fmt(numbers[i * chunk:(i + 1) * chunk]))
            say(f"  {label:<22} {' '.join(pieces)}")

    say(f"{total} numbers -> {world_size} chunks of {chunk} "
        f"({chunk * ITEM} bytes each)")
    show("my chunks to start")

    # -------------------------------------------------------------------
    # Lap 1: reduce-scatter. Add what arrives.
    # -------------------------------------------------------------------
    # On step 0 I send my own chunk (index = rank) and receive the one behind it.
    # Every step both indexes walk one place backwards round the ring, so each
    # chunk keeps moving forward and picking up another node's numbers.
    for step in range(world_size - 1):
        send_index = (rank - step) % world_size
        recv_index = (rank - step - 1) % world_size
        say(f"lap 1 step {step}: send chunk {send_index}, "
            f"add into chunk {recv_index}")

        send_at = send_index * chunk
        recv_at = recv_index * chunk

        # A slice of a 1-D array is a contiguous view, so .tobytes() gives the
        # raw doubles straight out - there is no conversion step.
        outgoing = numbers[send_at:send_at + chunk]

        # frombuffer reads the bytes that arrived as doubles in place, with no
        # copy and no decoding, because they are already exactly that. The array
        # it hands back is read-only, which is all we need - we only read it.
        theirs = np.frombuffer(link.exchange(outgoing.tobytes()),
                               dtype=np.float64)

        if tracing():
            mine = numbers[recv_at:recv_at + chunk].copy()
            say(f"  sent {fmt(outgoing)} to rank {link.next_rank}, "
                f"got {fmt(theirs)} from rank {link.prev_rank}")

        # This is the whole "reduce": my chunk plus theirs, number by number.
        # One line, but it is chunk separate additions, same as a loop over
        # range(chunk) would do and in the same order.
        numbers[recv_at:recv_at + chunk] += theirs

        if tracing():
            say(f"  chunk {recv_index}: {fmt(mine)} + {fmt(theirs)} "
                f"= {fmt(numbers[recv_at:recv_at + chunk])}")
        show(f"after lap 1 step {step}")

    # Chunk (rank + 1) is now finished on this node: the total of that chunk
    # from every node. Everybody holds exactly one finished chunk, and no two
    # nodes hold the same one.
    say(f"end of lap 1: chunk {(rank + 1) % world_size} is now the total from "
        f"all {world_size} nodes - and I am the only one who has it")

    # -------------------------------------------------------------------
    # Lap 2: all-gather. Overwrite what arrives.
    # -------------------------------------------------------------------
    # The same walk, shifted one place forward so it starts from the chunk we
    # just finished. Nothing is added now - these chunks are done, they only
    # need copying to everyone else.
    for step in range(world_size - 1):
        send_index = (rank - step + 1) % world_size
        recv_index = (rank - step) % world_size
        say(f"lap 2 step {step}: send chunk {send_index}, "
            f"store into chunk {recv_index}")

        send_at = send_index * chunk
        recv_at = recv_index * chunk
        outgoing = numbers[send_at:send_at + chunk]

        theirs = np.frombuffer(link.exchange(outgoing.tobytes()),
                               dtype=np.float64)

        numbers[recv_at:recv_at + chunk] = theirs

        if tracing():
            say(f"  sent {fmt(outgoing)} to rank {link.next_rank}, "
                f"got {fmt(theirs)} from rank {link.prev_rank} "
                f"-> chunk {recv_index} (overwrite, no adding)")
        show(f"after lap 2 step {step}")

    return numbers[:total]      # drop the padding zeros
