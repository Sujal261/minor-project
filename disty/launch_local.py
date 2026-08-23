"""
launch_local.py - start several nodes on this one machine, for testing.

    python launch_local.py --nodes 3
    python launch_local.py --nodes 3 --script train_ring.py
    python launch_local.py --nodes 3 --script trace_ring.py

Each node becomes its own process with its own sockets, so the ring is
completely real - only the network is fake, and the algorithm cannot tell the
difference.

This works on Windows, Linux and macOS with no changes and no flags. It picks
how the nodes talk to each other by asking Python what this computer supports:

  - Linux and macOS get local sockets (files in the system temp folder). Those
    never touch the network stack, so they are faster and immune to whatever the
    network is doing. This matters on WSL2, which routes 127.0.0.1 through
    Windows and drops packets under load - see new_socket() in ring.py.
  - Windows has no local sockets, so it gets 127.0.0.1 instead. Real Windows has
    real loopback, so this is perfectly fine there.

--tcp forces 127.0.0.1 even where local sockets exist.

The output of all the nodes is interleaved, so lines arrive in a jumbled order.
That is normal - every line says which rank it came from.
"""

import argparse
import os
import subprocess
import sys
import tempfile

import ring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=3,
                        help="how many nodes to start")
    parser.add_argument("--script", default="run.py",
                        help="what each node runs: run.py for the correctness "
                             "tests, train_ring.py for training, trace_ring.py "
                             "for a number-by-number trace")
    parser.add_argument("--tcp", action="store_true",
                        help="use 127.0.0.1 even if local sockets are available")
    parser.add_argument("--base-port", type=int, default=29600,
                        help="with TCP: the first port to use")
    parser.add_argument("--quiet", action="store_true",
                        help="do not print every ring step")
    parser.add_argument("--big-mb", type=int, default=None,
                        help="passed to run.py: size of the big test")
    parser.add_argument("--steps", type=int, default=None,
                        help="passed to train_ring.py: number of steps")
    args = parser.parse_args()

    # Build the peers list. Position in the list is the rank, and every node
    # gets this exact same string.
    #
    # Local sockets if this computer has them and the user did not ask for TCP.
    # tempfile.gettempdir() instead of a hardcoded "/tmp" so the path is right
    # wherever we are running.
    if args.tcp or not ring.local_sockets_work():
        peers = ",".join(f"127.0.0.1:{args.base_port + r}"
                         for r in range(args.nodes))
        kind = "127.0.0.1 (TCP)"
    else:
        folder = tempfile.gettempdir()
        names = [os.path.join(folder, f"disty-{r}.sock")
                 for r in range(args.nodes)]
        for name in names:              # clear out any crashed run
            if os.path.exists(name):
                os.unlink(name)
        peers = ",".join(names)
        kind = "local sockets"

    print(f"[launch] starting {args.nodes} nodes running {args.script}",
          flush=True)
    print(f"[launch] talking over {kind}", flush=True)
    print(f"[launch] peers: {peers}\n", flush=True)

    # Start every node as a separate process.
    processes = []
    for rank in range(args.nodes):
        command = [sys.executable, args.script,
                   "--rank", str(rank), "--peers", peers]
        if args.quiet:
            command.append("--quiet")
        if args.big_mb is not None:
            command += ["--big-mb", str(args.big_mb)]
        if args.steps is not None:
            command += ["--steps", str(args.steps)]
        processes.append(subprocess.Popen(command))

    # Wait for them all to finish and collect their exit codes.
    exit_codes = []
    for process in processes:
        exit_codes.append(process.wait())

    print()
    if all(code == 0 for code in exit_codes):
        print(f"[launch] all {args.nodes} nodes finished cleanly", flush=True)
        return 0

    for rank, code in enumerate(exit_codes):
        if code != 0:
            print(f"[launch] rank {rank} failed with exit code {code}",
                  flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
