import argparse
import json
import os
import resource
import signal
import threading
import time

import yaml

from .db import Pool
from .http import App, HTTPError, Upstream, make_server, serve_in_thread
from .log import Logger
from .metrics import Registry


def rss_bytes() -> int:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def resolve_config_file(home: str, name: str, override: str | None) -> str:
    default = os.path.realpath(os.path.join(home, "etc", f"{name}.yaml"))
    if not override:
        return default
    path = os.path.realpath(os.path.join(home, override))
    archive = os.path.realpath(os.path.join(home, "etc", "archive"))
    if path == default or os.path.dirname(path) == archive:
        return path
    raise SystemExit(
        f"NW_CONFIG_FILE must be etc/{name}.yaml or a file in etc/archive/: {override}"
    )


def read_token(home: str) -> str | None:
    try:
        with open(os.path.join(home, "run", "internal.token")) as f:
            return f.read().strip() or None
    except OSError:
        return None


class FlagsClient:
    def __init__(self, svc: "Service", url: str | None, interval_s: float, defaults: dict):
        self.svc = svc
        self.url = url
        self.interval_s = interval_s
        self.defaults = dict(defaults or {})
        self.values = dict(self.defaults)
        self.source = "local_defaults"
        self.last_ok = 0.0
        self._up = None
        if url:
            self._up = Upstream(svc.name, "flags", {"url": url, "timeout_ms": 500, "pool_size": 1})
        svc.metrics.gauge_fn("flags_source_remote", lambda: 1 if self.source == "remote" else 0)

    def fetch_once(self) -> bool:
        if self._up is None:
            return False
        try:
            r = self._up.get(f"/flags/{self.svc.name}")
            if not r.ok:
                raise ValueError(f"status {r.status}")
            values = r.json().get("flags", {})
        except Exception as e:
            if self.source == "remote" or self.last_ok == 0.0:
                self.svc.log.warn(
                    "flags service unreachable, using "
                    + ("cached values" if self.source == "remote" else "local defaults"),
                    error=str(e)[:120],
                )
            if self.source != "remote":
                self.source = "local_defaults"
            else:
                self.source = "cached"
            return False
        merged = {**self.defaults, **values}
        if merged != self.values and self.source == "remote":
            changed = sorted(k for k in merged if merged.get(k) != self.values.get(k))
            self.svc.log.info("flags updated", changed=",".join(changed))
        self.values = merged
        self.source = "remote"
        self.last_ok = time.time()
        return True

    def loop(self):
        while not self.svc.stopping.is_set():
            self.fetch_once()
            self.svc.stopping.wait(self.interval_s)

    def get(self, name, default=None):
        return self.values.get(name, default)


class Service:
    def __init__(self, name: str, argv=None):
        ap = argparse.ArgumentParser()
        ap.add_argument("--home", default=os.environ.get("NW_HOME", "."))
        args = ap.parse_args(argv)
        self.name = name
        self.home = os.path.abspath(args.home)
        self.config = self.load_config()
        log_cfg = self.config.get("log", {})
        self.log = Logger(name, self.path(f"logs/{name}.log"), log_cfg.get("level", "info"))
        self.metrics = Registry()
        self.app = App(name, self.log, self.metrics)
        self.internal_token = read_token(self.home)
        self.app.internal_token = self.internal_token
        self.stopping = threading.Event()
        self.started_at = time.time()
        self.ready_checks: list = []
        self.background: list = []
        self.flags: FlagsClient | None = None
        build = self.config.get("build", {})
        self.version = str(build.get("version", "0.0.0"))
        self.build_flags = dict(build.get("flags", {}) or {})
        self._install_builtin_routes()
        self.metrics.gauge_fn("process_resident_memory_bytes", rss_bytes)
        self.metrics.gauge_fn("process_uptime_seconds", lambda: time.time() - self.started_at)
        self.metrics.gauge_fn("log_write_errors_total", lambda: self.log.write_errors)

    def path(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(self.home, rel)

    def load_config(self) -> dict:
        path = resolve_config_file(self.home, self.name, os.environ.get("NW_CONFIG_FILE"))
        self.config_file = os.path.relpath(path, self.home)
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def upstream(self, name: str, internal: bool = True) -> Upstream:
        cfg = dict(self.config["upstreams"][name])
        if cfg.get("ca_file"):
            cfg["ca_file"] = self.path(cfg["ca_file"])
        token = self.internal_token if internal else None
        return Upstream(self.name, name, cfg, self.metrics, self.log, token=token)

    def database(self, name: str, key: str = "db") -> Pool:
        from northwind.schema import REQUIRED, schema_version

        c = self.config.get(key, {})
        pool = Pool(
            self.path(c["path"]),
            size=int(c.get("pool_size", 4)),
            acquire_timeout_ms=int(c.get("acquire_timeout_ms", 1000)),
            busy_timeout_ms=int(c.get("busy_timeout_ms", 2000)),
            metrics=self.metrics,
            name=name,
        )
        need = REQUIRED.get(name, 0)

        def check():
            v = schema_version(pool)
            return v >= need, f"{name} schema version {v} < required {need}"

        self.ready_checks.append(check)
        return pool

    def env(self, name: str, default: str | None = None) -> str | None:
        return os.environ.get(name, default)

    def use_flags(self):
        fcfg = self.config.get("flags", {})
        self.flags = FlagsClient(
            self, fcfg.get("url"), float(fcfg.get("poll_interval_s", 2)), fcfg.get("defaults", {})
        )
        self.flags.fetch_once()
        self.add_background(self.flags.loop)

    def flag(self, name, default=None):
        if self.flags is None:
            return default
        return self.flags.get(name, default)

    def add_background(self, fn, *args):
        self.background.append((fn, args))

    def every(self, interval_s: float, fn, name: str):
        def loop():
            while not self.stopping.wait(interval_s):
                try:
                    fn()
                except Exception as e:
                    self.log.error(
                        "background job failed", job=name, error=f"{type(e).__name__}: {e}"[:200]
                    )

        self.add_background(loop)

    def _install_builtin_routes(self):
        @self.app.route("GET", "/health")
        def health(req):
            return 200, {"status": "ok", "service": self.name}

        @self.app.route("GET", "/ready")
        def ready(req):
            failing = []
            for check in self.ready_checks:
                try:
                    ok, reason = check()
                except Exception as e:
                    ok, reason = False, f"{type(e).__name__}: {e}"
                if not ok:
                    failing.append(reason)
            if failing:
                return 503, {"ready": False, "reasons": failing}
            return 200, {"ready": True}

        @self.app.route("GET", "/metrics")
        def metrics(req):
            return 200, self.metrics.render()

        @self.app.route("GET", "/version")
        def version(req):
            return 200, {"service": self.name, "version": self.version, "build": self.build_flags}

    def run(self):
        listen = self.config.get("listen", {})
        host = listen.get("host", "127.0.0.1")
        port = int(listen["port"])
        tls = listen.get("tls")
        if tls:
            tls = {k: self.path(v) for k, v in tls.items()}
        srv = make_server(self.app, host, port, tls)

        def stop(signum, frame):
            self.stopping.set()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        sup = os.environ.get("NW_SUPERVISOR_PID")
        if sup:

            def orphan_watch():
                while not self.stopping.wait(1.0):
                    if os.getppid() != int(sup):
                        self.stopping.set()

            threading.Thread(target=orphan_watch, daemon=True).start()
        noise_s = float(self.config.get("log", {}).get("noise_interval_s", 8))
        if noise_s > 0 and self.name not in ("telemetry", "trafficgen", "provider"):
            from .noise import emitter

            self.every(noise_s, emitter(self), "log_noise")
        for fn, args in self.background:
            threading.Thread(target=fn, args=args, daemon=True).start()
        serve_in_thread(srv)
        self.log.info(
            "service started",
            config=self.config_file,
            version=self.version,
            port=port,
            build=json.dumps(self.build_flags, separators=(",", ":")),
        )
        self.stopping.wait()
        self.log.info("service stopping")
        srv.shutdown()
        srv.server_close()


def require(cond, status, error, **extra):
    if not cond:
        raise HTTPError(status, error, **extra)
