"""Start, stop and clean up one Northwind deployment ("stack") for an episode.

Each stack gets its own directory and its own block of ports. A port block is claimed by
holding an flock on a lock file, so concurrent stacks (threads or processes) never
collide, and a crashed owner releases its block automatically.

    with Stack(seed=3) as s:      # create + start, stop + delete on exit
        s.ports["edge"]

CLI (run inside the dev container or on a Linux VM, never on a macOS host):
    python -m oncallgym.stack up | probe | parallel N | clean
"""

import contextlib
import fcntl
import http.client
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

import northwind
from northwind.provision import provision
from northwind.topology import BLOCK_SIZE

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(northwind.__file__)))
BASE_PORT = int(os.environ.get("ONCALLGYM_BASE_PORT", 21000))
MAX_BLOCKS = 200


def default_root() -> str:
    return os.environ.get("ONCALLGYM_ROOT", os.path.join(tempfile.gettempdir(), "oncallgym"))


class HostNotAllowed(RuntimeError):
    pass


def assert_contained():
    """Services and agent commands must never run directly on the owner's macOS host."""
    if platform.system() == "Darwin":
        raise HostNotAllowed(
            "refusing to start Northwind on a macOS host; run inside the dev "
            "container (make shell) or on a Linux VM"
        )


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


LOCK_DIR = os.environ.get(
    "ONCALLGYM_LOCK_DIR", os.path.join(tempfile.gettempdir(), "oncallgym-ports")
)


def _sweep_stale(owner: str):
    """The previous holder of this block died without cleaning up: kill whatever is left
    of its stack and delete its directories (spec: cleanup even on crash)."""
    try:
        info = json.loads(owner)
    except ValueError:
        return
    home, private = info.get("home"), info.get("private")
    if home:
        for pid in leftover_processes(home):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    for d in (home, private):
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def claim_port_block(home: str, private: str) -> tuple[int, int]:
    """Return (block index, lock fd). The fd must stay open while the stack lives. Locks
    are machine-wide (not per root) so stacks under different roots never share ports."""
    os.makedirs(LOCK_DIR, exist_ok=True)
    for i in range(MAX_BLOCKS):
        fd = os.open(os.path.join(LOCK_DIR, f"block-{i}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        previous = os.pread(fd, 4096, 0).decode(errors="replace").strip()
        if previous:
            _sweep_stale(previous)
        base = BASE_PORT + i * BLOCK_SIZE
        if all(_port_free(base + k) for k in range(BLOCK_SIZE)):
            os.ftruncate(fd, 0)
            os.pwrite(
                fd, json.dumps({"pid": os.getpid(), "home": home, "private": private}).encode(), 0
            )
            return i, fd
        os.close(fd)
    raise RuntimeError("no free port block")


def http_json(port: int, method: str, path: str, body=None, timeout=5.0, token=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Internal-Token"] = token
        c.request(method, path, body=data, headers=headers)
        r = c.getresponse()
        raw = r.read()
        return r.status, (json.loads(raw) if raw else {})
    finally:
        c.close()


class Stack:
    def __init__(self, seed: int = 0, root: str | None = None, name: str | None = None):
        self.seed = seed
        self.root = root or default_root()
        self.name = name or f"ep-{uuid.uuid4().hex[:10]}"
        self.home = os.path.join(self.root, "episodes", self.name)
        self.private = os.path.join(self.root, "private", self.name)
        self.block = None
        self._lock_fd = None
        self.ports: dict[str, int] = {}
        self.proc: subprocess.Popen | None = None
        self.timings: dict[str, float] = {}

    # lifecycle
    def create(self):
        t0 = time.time()
        self.block, self._lock_fd = claim_port_block(self.home, self.private)
        base = BASE_PORT + self.block * BLOCK_SIZE
        info = provision(self.home, self.private, base, self.seed)
        self.ports = info["ports"]
        self.timings["provision_s"] = round(time.time() - t0, 2)
        return self

    def start(self, timeout_s: float = 30.0, require_ready: bool = True):
        """Start the supervisor and wait for boot. With require_ready (default) also wait
        until every service is ready; a stack started with a fault may never be."""
        assert_contained()
        if not self.ports:
            self.create()
        t0 = time.time()
        log = open(os.path.join(self.home, "logs/supervisor.stdout"), "ab")
        # services run with cwd=home, so make the code importable whether or not it is installed
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [CODE_ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "northwind.supervisor", "--home", self.home],
            cwd=self.home,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        log.close()
        deadline = t0 + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"supervisor exited with {self.proc.returncode}:\n"
                    + self.log_tail("supervisor.stdout")
                )
            try:
                st, body = http_json(self.ports["supervisor"], "GET", "/services", timeout=1)
                if (
                    st == 200
                    and body["booted"]
                    and (not require_ready or all(s["ready"] for s in body["services"]))
                ):
                    self.timings["cold_start_s"] = round(time.time() - t0, 2)
                    return self
            except (OSError, http.client.HTTPException, ValueError):
                pass
            time.sleep(0.2)
        raise TimeoutError(
            f"stack not ready after {timeout_s}s: {self.not_ready()}\n"
            f"supervisor log tail:\n{self.log_tail('supervisor', 15)}"
        )

    def log_tail(self, name: str, n: int = 20) -> str:
        fname = name if "." in name else f"{name}.log"
        try:
            with open(os.path.join(self.home, "logs", fname), errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError as e:
            return f"(no log: {e})"

    def not_ready(self):
        try:
            _, body = http_json(self.ports["supervisor"], "GET", "/services", timeout=1)
            return {
                s["name"]: s["not_ready_reasons"] or s["state"]
                for s in body["services"]
                if not s["ready"]
            }
        except (OSError, http.client.HTTPException, ValueError) as e:
            return f"supervisor unreachable: {e}"

    @property
    def token(self) -> str | None:
        from northwind.lib.service import read_token

        return read_token(self.home)

    def call(self, service: str, method: str, path: str, body=None, timeout=10.0):
        """HTTP call to one of this stack's services with the internal token."""
        return http_json(self.ports[service], method, path, body, timeout=timeout, token=self.token)

    def services(self) -> list[dict]:
        _, body = http_json(self.ports["supervisor"], "GET", "/services")
        return body["services"]

    def restart(self, name: str):
        return self.call("supervisor", "POST", f"/services/{name}/restart", timeout=15)

    def restart_all(self):
        return self.call("supervisor", "POST", "/restart-all", timeout=90)

    def total_rss_mb(self) -> float:
        """RSS of the supervisor plus every service process."""
        from northwind.supervisor import rss_of

        total = rss_of(self.proc.pid) or 0
        for s in self.services():
            if s["pid"]:
                total += rss_of(s["pid"]) or 0
        return round(total / 1e6, 1)

    def stop(self):
        if self.proc is None:
            return
        pgid = self.proc.pid
        with contextlib.suppress(OSError, http.client.HTTPException, ValueError):
            self.call("supervisor", "POST", "/shutdown", timeout=2)
        try:
            self.proc.wait(15)
        except subprocess.TimeoutExpired:
            pass
        # whatever is left in the process group gets killed
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            time.sleep(0.5)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(5)
        self.proc = None

    def cleanup(self):
        self.stop()
        shutil.rmtree(self.home, ignore_errors=True)
        shutil.rmtree(self.private, ignore_errors=True)
        if self._lock_fd is not None:
            # lock files are not unlinked (that would race with a waiting claimer); mark the
            # block as cleanly released so the next claimer has nothing to sweep
            with contextlib.suppress(OSError):
                os.ftruncate(self._lock_fd, 0)
            os.close(self._lock_fd)
            self._lock_fd = None

    def __enter__(self):
        try:
            self.create()
            self.start()
        except BaseException:
            self.cleanup()
            raise
        return self

    def __exit__(self, *exc):
        self.cleanup()


def leftover_processes(root: str | None = None) -> list[int]:
    """PIDs of any Northwind processes whose command line references this root."""
    root = root or default_root()
    pids = []
    proc = "/proc"
    if not os.path.isdir(proc):
        return pids
    for d in os.listdir(proc):
        if not d.isdigit() or int(d) == os.getpid():
            continue
        try:
            with open(f"{proc}/{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if ("northwind.supervisor" in cmd or "northwind.services." in cmd) and root in cmd:
            pids.append(int(d))
    return pids


def clean(root: str | None = None) -> dict:
    """Kill every Northwind process under root, delete root, and reset port locks that no
    live stack holds."""
    root = root or default_root()
    killed = []
    for pid in leftover_processes(root):
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
    shutil.rmtree(root, ignore_errors=True)
    released = 0
    if os.path.isdir(LOCK_DIR):
        for name in os.listdir(LOCK_DIR):
            path = os.path.join(LOCK_DIR, name)
            fd = os.open(path, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # a live stack (possibly under another root) holds it
            finally:
                os.close(fd)
            os.unlink(path)
            released += 1
        with contextlib.suppress(OSError):
            os.rmdir(LOCK_DIR)
    return {"killed": killed, "removed": root, "stale_locks_removed": released}


def _print(obj):
    print(json.dumps(obj, indent=1, default=str))


def _cmd_up(args):
    s = Stack(seed=args.seed)
    try:
        s.create().start()
        print(f"stack {s.name} up in {s.timings['cold_start_s']}s, total RSS {s.total_rss_mb()} MB")
        print(f"home: {s.home}")
        for name, port in sorted(s.ports.items(), key=lambda kv: kv[1]):
            print(f"  {name:12s} 127.0.0.1:{port}")
        print("Ctrl-C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        s.cleanup()
        print("stopped and removed")


def _cmd_probe(args):
    from .probes import Probe

    with Stack(seed=args.seed) as s:
        time.sleep(args.warmup)
        result = Probe(s).run(journeys=args.journeys, window_s=args.window)
        result["cold_start_s"] = s.timings["cold_start_s"]
        result["total_rss_mb"] = s.total_rss_mb()
        result["restarts"] = {x["name"]: x["restarts"] for x in s.services() if x["restarts"]}
    result["leftover_processes"] = leftover_processes(s.root)
    result["home_removed"] = not os.path.exists(s.home)
    _print(result)
    sys.exit(0 if result["passed"] and not result["leftover_processes"] else 1)


def _cmd_parallel(args):
    import concurrent.futures as cf

    from .probes import Probe

    n = args.n
    stacks = [Stack(seed=i) for i in range(n)]
    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(n) as ex:
            list(ex.map(lambda s: s.create().start(timeout_s=60), stacks))
        all_up_s = round(time.time() - t0, 2)
        time.sleep(args.warmup)
        with cf.ThreadPoolExecutor(n) as ex:
            results = list(ex.map(lambda s: Probe(s).run(journeys=1, window_s=args.window), stacks))
        ports = [p for s in stacks for p in s.ports.values()]
        summary = {
            "stacks": n,
            "all_up_s": all_up_s,
            "cold_start_s": [s.timings["cold_start_s"] for s in stacks],
            "rss_mb": [s.total_rss_mb() for s in stacks],
            "port_collisions": len(ports) - len(set(ports)),
            "passed": [r["passed"] for r in results],
            "failures": [
                {k: v for k, v in r.items() if k != "journeys"} | {"journeys": r["journeys"]}
                for r in results
                if not r["passed"]
            ],
            "restarts": [sum(x["restarts"] for x in s.services()) for s in stacks],
        }
    finally:
        for s in stacks:
            s.cleanup()
    summary["leftover_processes"] = leftover_processes(stacks[0].root)
    _print(summary)
    ok = (
        all(summary["passed"])
        and not summary["port_collisions"]
        and not summary["leftover_processes"]
    )
    sys.exit(0 if ok else 1)


def _cmd_clean(args):
    _print(clean())
    left = leftover_processes()
    _print({"leftover_processes": left, "root_exists": os.path.exists(default_root())})


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="python -m oncallgym.stack")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up")
    up.add_argument("--seed", type=int, default=0)
    pr = sub.add_parser("probe")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--warmup", type=float, default=10)
    pr.add_argument("--journeys", type=int, default=3)
    pr.add_argument("--window", type=float, default=10)
    pa = sub.add_parser("parallel")
    pa.add_argument("n", type=int, nargs="?", default=6)
    pa.add_argument("--warmup", type=float, default=10)
    pa.add_argument("--window", type=float, default=10)
    sub.add_parser("clean")
    args = ap.parse_args(argv)
    {"up": _cmd_up, "probe": _cmd_probe, "parallel": _cmd_parallel, "clean": _cmd_clean}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
