"""Per-deployment process supervisor.

Starts services layer by layer (waiting for readiness between layers), restarts crashed
services with exponential backoff, kills services whose RSS exceeds their memory limit
(logged like a kernel OOM kill), and exposes a small control API:

  GET  /services                 status of every service
  POST /services/{name}/restart  stop + start one service (re-reads its config and env file)
  POST /restart-all
  POST /shutdown

All services run in the supervisor's process group, so killing the group removes
everything. The supervisor exits if its parent dies; services exit if the supervisor dies.
"""

import argparse
import http.client
import json
import os
import signal
import ssl
import subprocess
import sys
import threading
import time

import yaml

from northwind.lib.http import App, make_server, serve_in_thread
from northwind.lib.log import Logger
from northwind.lib.metrics import Registry

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_of(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * PAGE
    except OSError:
        pass
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return int(out) * 1024 if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def read_env_file(path: str) -> dict:
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"')
    except OSError:
        pass
    return env


class Proc:
    def __init__(self, name, cfg):
        self.name = name
        self.port = int(cfg["port"])
        self.memory_limit = int(cfg.get("memory_limit_mb", 150)) * 1_000_000
        self.popen: subprocess.Popen | None = None
        self.state = "stopped"  # starting, running, backoff, stopped
        self.restarts = 0
        self.oom_kills = 0
        self.last_exit = None
        self.started_at = None
        self.next_start = 0.0
        self.backoff = 0.0
        self.ready = False
        self.ready_reasons = None
        self.rss = None
        self.wanted = True


class Supervisor:
    def __init__(self, home: str):
        self.home = home
        with open(os.path.join(home, "etc/supervisor.yaml")) as f:
            self.cfg = yaml.safe_load(f)
        self.log = Logger("supervisor", os.path.join(home, "logs/supervisor.log"))
        self.procs = {n: Proc(n, c) for n, c in self.cfg["services"].items()}
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.parent = os.getppid()
        rs = self.cfg.get("restart", {})
        self.backoff_s = float(rs.get("backoff_s", 1))
        self.max_backoff_s = float(rs.get("max_backoff_s", 20))
        self.stable_after_s = float(rs.get("stable_after_s", 30))
        ca = os.path.join(home, "etc/tls/ca.pem")
        self.ssl_ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else None
        self.booted = False
        self.boot_seconds = None

    # process control
    def spawn(self, p: Proc):
        env = {k: v for k, v in os.environ.items() if not k.startswith("NW_")}
        env.update(read_env_file(os.path.join(self.home, f"etc/env/{p.name}.env")))
        env["NW_HOME"] = self.home
        env["NW_SUPERVISOR_PID"] = str(os.getpid())
        env["PYTHONUNBUFFERED"] = "1"
        out = open(os.path.join(self.home, f"logs/{p.name}.stdout"), "ab")
        p.popen = subprocess.Popen(
            [sys.executable, "-m", f"northwind.services.{p.name}", "--home", self.home],
            cwd=self.home,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        out.close()
        p.state = "starting"
        p.started_at = time.time()
        p.ready = False
        self.log.info("started service", service=p.name, pid=p.popen.pid)

    def terminate(self, p: Proc, grace_s: float = 3.0):
        if p.popen is None or p.popen.poll() is not None:
            p.popen = None
            return
        p.popen.terminate()
        try:
            p.popen.wait(grace_s)
        except subprocess.TimeoutExpired:
            p.popen.kill()
            p.popen.wait(2)
        p.popen = None

    def restart(self, name: str):
        with self.lock:
            p = self.procs[name]
            self.log.info("restart requested", service=name)
            p.wanted = False
            self.terminate(p)
            p.restarts += 1
            p.backoff = 0.0
            p.wanted = True
            self.spawn(p)

    def check_ready(self, p: Proc) -> bool:
        scheme_https = p.name == "auth"
        try:
            if scheme_https:
                c = http.client.HTTPSConnection(
                    "127.0.0.1", p.port, timeout=1.0, context=self.ssl_ctx
                )
            else:
                c = http.client.HTTPConnection("127.0.0.1", p.port, timeout=1.0)
            c.request("GET", "/ready")
            r = c.getresponse()
            body = r.read()
            c.close()
            p.ready = r.status == 200
            p.ready_reasons = None if p.ready else json.loads(body or b"{}").get("reasons")
        except (OSError, http.client.HTTPException, ValueError):
            p.ready = False
            p.ready_reasons = ["not responding"]
        return p.ready

    def monitor_once(self):
        now = time.time()
        with self.lock:
            for p in self.procs.values():
                if not p.wanted:
                    continue
                if p.popen is not None:
                    code = p.popen.poll()
                    if code is None:
                        rss = rss_of(p.popen.pid)
                        p.rss = rss
                        if rss and rss > p.memory_limit:
                            p.oom_kills += 1
                            self.log.error(
                                "Out of memory: killed process",
                                service=p.name,
                                pid=p.popen.pid,
                                rss_mb=round(rss / 1e6, 1),
                                limit_mb=round(p.memory_limit / 1e6),
                            )
                            p.popen.kill()
                            p.popen.wait(2)
                            code = -9
                        elif (
                            p.state == "running"
                            and p.backoff
                            and now - p.started_at > self.stable_after_s
                        ):
                            p.backoff = 0.0
                    if code is not None:
                        p.last_exit = code
                        p.popen = None
                        p.ready = False
                        p.backoff = (
                            min(self.max_backoff_s, p.backoff * 2) if p.backoff else self.backoff_s
                        )
                        p.next_start = now + p.backoff
                        p.state = "backoff"
                        self.log.error(
                            "service exited", service=p.name, code=code, restart_in_s=p.backoff
                        )
                elif p.state == "backoff" and now >= p.next_start:
                    p.restarts += 1
                    self.spawn(p)

    def ready_loop(self):
        while not self.stopping.wait(1.0):
            for p in list(self.procs.values()):
                if p.popen is not None and p.popen.poll() is None:
                    was = p.ready
                    self.check_ready(p)
                    if p.ready:
                        p.state = "running"
                    if was and not p.ready:
                        self.log.warn(
                            "service became not ready",
                            service=p.name,
                            reasons=";".join(p.ready_reasons or [])[:160],
                        )

    def monitor_loop(self):
        while not self.stopping.wait(0.5):
            if os.getppid() != self.parent:
                self.log.warn("parent process gone, shutting down")
                self.stopping.set()
                break
            self.monitor_once()

    def boot(self):
        t0 = time.time()
        timeout = float(self.cfg.get("layer_ready_timeout_s", 8))
        for layer in self.cfg["layers"]:
            with self.lock:
                for name in layer:
                    self.spawn(self.procs[name])
            deadline = time.time() + timeout
            pending = set(layer)
            while pending and time.time() < deadline and not self.stopping.is_set():
                time.sleep(0.1)
                for name in list(pending):
                    p = self.procs[name]
                    if p.popen is not None and self.check_ready(p):
                        p.state = "running"
                        pending.discard(name)
            if pending:
                self.log.warn(
                    "layer not ready before timeout, continuing", services=",".join(sorted(pending))
                )
        self.boot_seconds = round(time.time() - t0, 2)
        self.booted = True
        self.log.info("boot complete", seconds=self.boot_seconds)

    def shutdown(self):
        with self.lock:
            for p in self.procs.values():
                p.wanted = False
            for layer in reversed(self.cfg["layers"]):
                for name in layer:
                    p = self.procs[name]
                    if p.popen is not None and p.popen.poll() is None:
                        p.popen.terminate()
                deadline = time.time() + 3
                for name in layer:
                    p = self.procs[name]
                    if p.popen is None:
                        continue
                    try:
                        p.popen.wait(max(0.1, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        p.popen.kill()
                        p.popen.wait(2)
                    p.popen = None
                    p.state = "stopped"
        self.log.info("all services stopped")

    def status(self):
        now = time.time()
        out = []
        for p in self.procs.values():
            out.append(
                {
                    "name": p.name,
                    "state": p.state,
                    "ready": p.ready,
                    "pid": p.popen.pid if p.popen else None,
                    "restarts": p.restarts,
                    "oom_kills": p.oom_kills,
                    "last_exit": p.last_exit,
                    "rss_mb": round(p.rss / 1e6, 1) if p.rss else None,
                    "memory_limit_mb": round(p.memory_limit / 1e6),
                    "uptime_s": round(now - p.started_at, 1) if p.popen and p.started_at else None,
                    "not_ready_reasons": p.ready_reasons if not p.ready else None,
                }
            )
        return out

    def serve(self):
        app = App("supervisor", self.log, Registry())
        app.quiet_paths.add("/services")

        @app.route("GET", "/services")
        def services(req):
            return 200, {
                "booted": self.booted,
                "boot_seconds": self.boot_seconds,
                "services": self.status(),
            }

        @app.route("POST", "/services/{name}/restart")
        def restart(req):
            name = req.params["name"]
            if name not in self.procs:
                return 404, {"error": "unknown service"}
            self.restart(name)
            return 200, {"restarted": name}

        @app.route("POST", "/restart-all")
        def restart_all(req):
            for layer in self.cfg["layers"]:
                for name in layer:
                    self.restart(name)
            return 200, {"restarted": "all"}

        @app.route("POST", "/shutdown")
        def shutdown(req):
            threading.Thread(
                target=lambda: (time.sleep(0.1), self.stopping.set()), daemon=True
            ).start()
            return 200, {"stopping": True}

        @app.route("GET", "/health")
        def health(req):
            return 200, {"status": "ok"}

        srv = make_server(app, "127.0.0.1", int(self.cfg["listen"]["port"]))
        serve_in_thread(srv)
        return srv

    def run(self):
        signal.signal(signal.SIGTERM, lambda *a: self.stopping.set())
        signal.signal(signal.SIGINT, lambda *a: self.stopping.set())
        with open(os.path.join(self.home, "run/supervisor.pid"), "w") as f:
            f.write(str(os.getpid()))
        srv = self.serve()
        threading.Thread(target=self.monitor_loop, daemon=True).start()
        self.boot()
        threading.Thread(target=self.ready_loop, daemon=True).start()
        self.stopping.wait()
        srv.shutdown()
        self.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    args = ap.parse_args()
    Supervisor(os.path.abspath(args.home)).run()


if __name__ == "__main__":
    main()
