"""
run.py - run one node of the ring and check the all-reduce is correct.

ON ONE MACHINE (easiest way to try it)
    python launch_local.py --nodes 3

ON REAL DEVICES
Give every device the SAME --peers list and change only --rank:

    # on 10.0.0.1
    python run.py --rank 0 --peers 10.0.0.1:29600,10.0.0.2:29600

    # on 10.0.0.2
    python run.py --rank 1 --peers 10.0.0.1:29600,10.0.0.2:29600

The order of the list decides who is rank 0, rank 1 and so on.
"""

import argparse
import time
from array import array

from ring import connect_ring, parse_peers
from allreduce import all_reduce


def check(name, got, want, rank):
    """Compare a result against the answer we worked out by hand."""
    same_length = len(got) == len(want)
    close_enough = all(abs(g - w) < 1e-9 for g, w in zip(got, want))
    ok = same_length and close_enough

    print(f"[rank {rank}] {'PASS' if ok else 'FAIL'}  {name}", flush=True)
    if not ok:
        print(f"[rank {rank}]       got  {list(got)[:12]}", flush=True)
        print(f"[rank {rank}]       want {list(want)[:12]}", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True,
                        help="which node this is: 0, 1, 2 ...")
    parser.add_argument("--peers", required=True,
                        help="addresses in rank order, comma separated: "
                             "host:port for TCP, or a path for local sockets")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print every ring step")
    parser.add_argument("--big-mb", type=int, default=45,
                        help="size of the final big test, in MB")
    args = parser.parse_args()

    peers = parse_peers(args.peers)
    rank = args.rank
    world_size = len(peers)
    show_steps = not args.quiet

    print(f"[rank {rank}] starting, {world_size} nodes in the ring", flush=True)
    link = connect_ring(rank, peers, verbose=show_steps)

    # Shorthand so the tests below stay readable.
    def reduce_it(values, average=False, verbose=False):
        return all_reduce(values, rank, world_size, link,
                          average=average, verbose=verbose)

    # If every node sends (rank + 1), the total is 1 + 2 + ... + world_size.
    expected_sum = world_size * (world_size + 1) / 2

    try:
        # --- 1. a worked example, printing every step ---------------------
        print(f"\n[rank {rank}] === example: adding up simple lists ===",
              flush=True)

        values = [float(rank + 1)] * 6
        print(f"[rank {rank}] my list before: {values}", flush=True)

        result = reduce_it(values, average=False, verbose=show_steps)
        print(f"[rank {rank}] after all-reduce: {list(result)}", flush=True)
        check("sum of all lists", result, [expected_sum] * 6, rank)

        # The average is what training actually wants.
        result = reduce_it([float(rank + 1)] * 6, average=True)
        check("average of all lists", result,
              [expected_sum / world_size] * 6, rank)

        # --- 2. a different number in every slot -------------------------
        # The test above would still pass if the chunks came back shuffled,
        # because every element is the same. This one would not, so this is the
        # test that proves the chunk indexing is right.
        print(f"\n[rank {rank}] === checking the chunk order ===", flush=True)

        values = [float((rank + 1) * i) for i in range(10)]
        result = reduce_it(values)
        check("each slot summed separately", result,
              [i * expected_sum for i in range(10)], rank)

        # --- 3. lengths that do not divide evenly ------------------------
        # 10 numbers across 3 nodes does not divide, so allreduce.py pads up to
        # 12 and cuts the padding off afterwards. A list SHORTER than the node
        # count is the nastiest case - most chunks are then pure padding.
        print(f"\n[rank {rank}] === checking awkward lengths ===", flush=True)

        lengths = {1, world_size - 1, world_size + 1, world_size * 2 + 1,
                   7, 100}
        for length in sorted(lengths):
            if length < 1:
                continue
            result = reduce_it([float(rank + 1)] * length)
            check(f"length {length}", result, [expected_sum] * length, rank)

        # --- 4. a realistic amount of data -------------------------------
        # Everything above is tiny, and tiny is exactly what hides the bug this
        # code is built to avoid. A small block vanishes into the socket buffer,
        # so even a broken send-then-receive would work by luck. Once a block is
        # bigger than that buffer the send cannot finish until the other side
        # reads, and a ring where everyone sends first locks up solid. So push
        # through a gradient-sized amount of data - this is the test that matters.
        big_length = args.big_mb * 1_000_000 // 8      # 8 bytes per double
        print(f"\n[rank {rank}] === checking {args.big_mb} MB "
              f"({big_length:,} numbers) ===", flush=True)

        # Built straight as an array, never as a Python list - a list of five
        # million floats would be five million separate objects.
        big = array("d", [float(rank + 1)]) * big_length

        start = time.time()
        result = reduce_it(big)
        seconds = time.time() - start

        # Checking five million numbers one by one is slow and tells us nothing
        # extra, so sample the ends and the middle. Test 2 is what proves the
        # ordering.
        spots = [0, 1, big_length // 2, big_length - 2, big_length - 1]
        check(f"{args.big_mb} MB reduced", [result[i] for i in spots],
              [expected_sum] * len(spots), rank)
        check(f"{args.big_mb} MB length kept", [len(result)], [big_length], rank)

        # Each node moves 2*(N-1)/N of its buffer, in each direction.
        moved = 2 * (world_size - 1) / world_size * args.big_mb
        print(f"[rank {rank}]       took {seconds:.2f}s "
              f"({moved:.0f} MB each way, {moved / seconds:.0f} MB/s)",
              flush=True)

        print(f"\n[rank {rank}] all done", flush=True)

    finally:
        link.close()


if __name__ == "__main__":
    main()
