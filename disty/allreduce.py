"""
allreduce.py - the ring all-reduce itself. No sockets in here at all.

WHAT IT DOES
Every node starts with its own list of numbers, all the same length. When this
finishes, every node holds the element-wise total (or average) of everybody's
lists. In training those numbers are the gradients: each node worked out a
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

WHY array('d') AND NOT A LIST
An array holds raw 8-byte doubles back to back, so handing it to a socket is
just .tobytes() - no encoding step, nothing hidden. A list of the same numbers
would be a list of pointers to millions of separate float objects, and turning
that into bytes would copy and rebuild every one of them. 'd' is a double,
which is what a Python float already is, so no precision is lost.
"""

from array import array


ITEM = 8            # bytes per number, because 'd' is an 8-byte double

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
    number of values. Returns an array('d') of the totals, or of the averages if
    average=True.

    `link` is the Link from ring.connect_ring - our two neighbours.

    All this does is walk the numbers in slices and hand each one to
    reduce_one_slice below. Every node slices identically, so every node is
    always working on the same slice as everybody else.
    """
    numbers = array("d", values)

    # One node is not a ring - nobody to talk to and nothing to add.
    if world_size == 1:
        return numbers

    result = array("d")
    for start in range(0, len(numbers), SLICE):
        part = numbers[start:start + SLICE]
        if verbose:
            print(f"[rank {rank}] slice {start}..{start + len(part)}",
                  flush=True)
        result.extend(reduce_one_slice(part, rank, world_size, link, verbose))

    # Turn the totals into averages. Done once at the end rather than before
    # sending, so there is one division pass instead of one per step.
    if average:
        for i in range(len(result)):
            result[i] = result[i] / world_size

    return result


def reduce_one_slice(numbers, rank, world_size, link, verbose=False):
    """
    The ring all-reduce on one slice. `numbers` is an array('d'); returns a new
    array('d') of the totals across all nodes.
    """
    total = len(numbers)

    # Pad with zeros so the slice divides into N equal chunks.
    numbers = numbers[:]        # our own copy - do not disturb the caller's
    numbers.extend([0.0] * (pad_length(total, world_size) - total))
    chunk = len(numbers) // world_size

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
        outgoing = numbers[send_at:send_at + chunk]

        theirs = array("d")
        theirs.frombytes(link.exchange(outgoing.tobytes()))

        if tracing():
            mine = numbers[recv_at:recv_at + chunk]
            say(f"  sent {fmt(outgoing)} to rank {link.next_rank}, "
                f"got {fmt(theirs)} from rank {link.prev_rank}")

        # This is the whole "reduce": my chunk plus theirs, number by number.
        for i in range(chunk):
            numbers[recv_at + i] = numbers[recv_at + i] + theirs[i]

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

        theirs = array("d")
        theirs.frombytes(link.exchange(outgoing.tobytes()))

        numbers[recv_at:recv_at + chunk] = theirs

        if tracing():
            say(f"  sent {fmt(outgoing)} to rank {link.next_rank}, "
                f"got {fmt(theirs)} from rank {link.prev_rank} "
                f"-> chunk {recv_index} (overwrite, no adding)")
        show(f"after lap 2 step {step}")

    return numbers[:total]      # drop the padding zeros
