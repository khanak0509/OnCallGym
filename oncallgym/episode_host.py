"""Run one Episode in its own process and serve it over local HTTP (episode_server).

    python -m oncallgym.episode_host --family F --variant V --seed S --ready-file PATH

Started by an adapter (e.g. verifiers' runtime.run_background). The host writes
{"url": ...} to --ready-file once the stack is up and the fault injected. On SIGTERM
(adapter teardown) or when its parent dies, it tears the stack down, so cleanup does not
depend on the adapter's own hooks running.
"""

import argparse
import json
import os
import signal
import threading

from .episode import Episode
from .episode_server import EpisodeServer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--variant", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--trajectory-file", default=None)
    args = ap.parse_args(argv)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    parent = os.getppid()
    ep = Episode(args.family, args.variant, args.seed)
    server = None
    try:
        messages = ep.start()
        server = EpisodeServer(ep).start()
        tmp = args.ready_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"url": server.url, "messages": messages, "pid": os.getpid()}, f)
        os.replace(tmp, args.ready_file)
        while not stop.wait(1.0):
            if os.getppid() != parent:
                break
    except Exception as e:
        with open(args.ready_file, "w") as f:
            json.dump({"error": f"{type(e).__name__}: {e}"[:500]}, f)
    finally:
        if args.trajectory_file:
            with open(args.trajectory_file, "w") as f:
                json.dump(ep.trajectory, f, default=str)
        if server:
            server.stop()
        ep.close()


if __name__ == "__main__":
    main()
