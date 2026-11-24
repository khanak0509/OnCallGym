import argparse
import json
import os
import signal
import threading

from .episode import Episode
from .episode_server import EpisodeServer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-file", required=True)
    ap.add_argument("--ready-file", required=True)
    ap.add_argument("--trajectory-file", default=None)
    args = ap.parse_args(argv)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    parent = os.getppid()
    with open(args.task_file) as f:
        task = json.load(f)
    os.unlink(args.task_file)
    ep = Episode(task["family"], task["variant"], task.get("seed"))
    server = None
    try:
        messages = ep.start()
        server = EpisodeServer(ep).start()
        tmp = args.ready_file + ".tmp"
        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(
                {
                    "url": server.url,
                    "token": server.token,
                    "messages": messages,
                    "pid": os.getpid(),
                },
                f,
            )
        os.replace(tmp, args.ready_file)
        while not stop.wait(1.0):
            if os.getppid() != parent:
                break
    except Exception as e:
        with open(args.ready_file, "w") as f:
            json.dump({"error": f"{type(e).__name__}: {e}"[:500]}, f)
    finally:
        if args.trajectory_file:
            os.makedirs(os.path.dirname(args.trajectory_file) or ".", mode=0o700, exist_ok=True)
            fd = os.open(args.trajectory_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(ep.trajectory, f, default=str)
        if server:
            server.stop()
        ep.close()


if __name__ == "__main__":
    main()
