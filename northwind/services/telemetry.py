import glob
import os
import re
import threading
import time
from collections import OrderedDict, deque

from northwind.lib.http import Upstream, UpstreamError
from northwind.lib.log import parse_line
from northwind.lib.metrics import parse_text
from northwind.lib.service import Service, require

BUCKET_RE = re.compile(r"_bucket$")


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def p_quantile(buckets: list[tuple[float, float]], q: float):
    if not buckets or buckets[-1][1] <= 0:
        return None
    target = q * buckets[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in buckets:
        if c >= target:
            if le == float("inf"):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (target - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le


def main():
    svc = Service("telemetry")
    cfg = svc.config
    window_s = float(cfg.get("window_s", 30))
    targets = {}
    for name, tcfg in cfg["targets"].items():
        tc = {"url": tcfg["url"], "timeout_ms": 800, "pool_size": 1}
        if tcfg.get("ca_file"):
            tc["ca_file"] = svc.path(tcfg["ca_file"])
        targets[name] = Upstream("telemetry", name, tc)
    samples: dict[str, deque] = {n: deque(maxlen=int(window_s * 3)) for n in targets}
    up_state: dict[str, dict] = {
        n: {"up": False, "ready": False, "last_error": None} for n in targets
    }
    spans: OrderedDict[str, list] = OrderedDict()
    max_traces = int(cfg.get("max_traces", 5000))
    lock = threading.Lock()
    log_counts: dict[str, deque] = {}
    tails: dict[str, dict] = {}
    disk = {"bytes": 0, "quota_bytes": cfg.get("disk_quota_bytes")}

    def scrape():
        now = time.time()
        for name, up in targets.items():
            st = up_state[name]
            try:
                r = up.get("/metrics")
                parsed = parse_text(r.body.decode(errors="replace")) if r.ok else []
                st["up"] = r.ok
                rr = up.get("/ready")
                st["ready"] = rr.ok
                st["ready_reasons"] = None if rr.ok else rr.json().get("reasons")
                st["last_error"] = None
            except UpstreamError as e:
                st.update(up=False, ready=False, last_error=e.kind)
                parsed = None
            with lock:
                samples[name].append((now, parsed))

    def tail_logs():
        for path in glob.glob(svc.path("logs/*.log")):
            name = os.path.basename(path)[:-4]
            try:
                stt = os.stat(path)
            except OSError:
                continue
            t = tails.setdefault(path, {"ino": stt.st_ino, "pos": 0})
            if t["ino"] != stt.st_ino or stt.st_size < t["pos"]:
                t.update(ino=stt.st_ino, pos=0)
            if stt.st_size == t["pos"]:
                continue
            with open(path, "rb") as f:
                f.seek(t["pos"])
                chunk = f.read(min(stt.st_size - t["pos"], 4_000_000))
            last_nl = chunk.rfind(b"\n")
            if last_nl < 0:
                continue
            t["pos"] += last_nl + 1
            now = time.time()
            counts = log_counts.setdefault(name, deque(maxlen=2000))
            for raw in chunk[: last_nl + 1].decode(errors="replace").splitlines():
                rec = parse_line(raw)
                lvl = rec.get("level")
                if lvl in ("warn", "error"):
                    counts.append((now, lvl))
                rid = rec.get("req")
                if not rid:
                    continue
                with lock:
                    lst = spans.get(rid)
                    if lst is None:
                        lst = spans[rid] = []
                        if len(spans) > max_traces:
                            spans.popitem(last=False)
                    if len(lst) < 60:
                        lst.append(rec)

    def disk_usage():
        disk["bytes"] = dir_size(svc.home)

    svc.every(float(cfg.get("scrape_interval_s", 2)), scrape, "scrape")
    svc.every(float(cfg.get("log_interval_s", 1)), tail_logs, "tail_logs")
    svc.every(10.0, disk_usage, "disk_usage")

    def summarize(name):
        with lock:
            window = [(t, s) for t, s in samples[name] if s is not None]
        st = up_state[name]
        out = {"service": name, "up": st["up"], "ready": st["ready"]}
        if st.get("ready_reasons"):
            out["not_ready_reasons"] = st["ready_reasons"]
        if not window:
            return out
        t_last, last = window[-1]
        t_first, first = window[0]
        for t, s in window:
            if t_last - t <= window_s:
                t_first, first = t, s
                break

        def total(sample, name_, pred=lambda labels: True):
            return sum(v for n, labels, v in sample if n == name_ and pred(labels))

        dt = max(t_last - t_first, 1e-6)
        req_now = total(last, "http_requests_total")
        req_then = total(first, "http_requests_total")
        err = lambda lab: lab.get("status", "").startswith("5")  # noqa: E731
        err_now = total(last, "http_requests_total", err)
        err_then = total(first, "http_requests_total", err)
        dreq = max(req_now - req_then, 0)
        out["rps"] = round(dreq / dt, 2) if len(window) > 1 else None
        out["error_rate"] = round((err_now - err_then) / dreq, 3) if dreq else 0.0
        le_now, le_then = {}, {}
        for n, lab, v in last:
            if n == "http_request_duration_ms_bucket":
                le_now[lab["le"]] = le_now.get(lab["le"], 0) + v
        for n, lab, v in first:
            if n == "http_request_duration_ms_bucket":
                le_then[lab["le"]] = le_then.get(lab["le"], 0) + v
        buckets = sorted(
            ((float(le), le_now[le] - le_then.get(le, 0)) for le in le_now), key=lambda x: x[0]
        )
        p95 = p_quantile(buckets, 0.95) if dreq else None
        out["p95_ms"] = round(p95, 1) if p95 is not None else None
        g = {}
        for n, lab, v in last:
            if n == "process_resident_memory_bytes":
                out["rss_mb"] = round(v / 1e6, 1)
            elif n == "circuit_breaker_state":
                g.setdefault("breakers", {})[lab["upstream"]] = {
                    0: "closed",
                    1: "half_open",
                    2: "open",
                }.get(int(v), v)
            elif n in ("db_pool_in_use", "db_pool_size"):
                g.setdefault("db_pool", {}).setdefault(lab["db"], {})[n[8:]] = int(v)
            elif n in ("upstream_pool_in_use", "upstream_pool_size"):
                if n == "upstream_pool_in_use" and v > 0:
                    g.setdefault("http_pools_busy", {})[lab["upstream"]] = int(v)
            elif n in ("broker_queue_depth", "broker_dlq_size", "broker_inflight"):
                g.setdefault("queues", {}).setdefault(lab["queue"], {})[n[7:]] = int(v)
            elif n in (
                "cache_used_bytes",
                "cache_max_bytes",
                "cache_rejected_total",
                "cache_evictions_total",
            ):
                g.setdefault("cache", {})[n[6:]] = int(v)
            elif n == "flags_source_remote":
                g["flags"] = "remote" if v else "local_defaults"
            elif n == "db_pool_timeouts_total" and v:
                g.setdefault("db_pool", {}).setdefault(lab["db"], {})["timeouts"] = int(v)
        for n, lab, v in last:
            if n == "upstream_requests_total" and lab.get("outcome") != "ok":
                before = total(first, n, lambda lb, lab=lab: lb == lab)
                if v - before > 0:
                    g.setdefault("upstream_errors", {}).setdefault(lab["upstream"], {})[
                        lab["outcome"]
                    ] = int(v - before)
        out.update(g)
        now = time.time()
        recent = [lvl for t, lvl in log_counts.get(name, ()) if now - t <= window_s]
        out["log_errors"] = recent.count("error")
        out["log_warnings"] = recent.count("warn")
        return out

    @svc.app.route("GET", "/summary")
    def summary(req):
        return 200, {
            "window_s": window_s,
            "services": [summarize(n) for n in targets],
            "disk_bytes": disk["bytes"],
            "disk_quota_bytes": disk["quota_bytes"],
        }

    @svc.app.route("GET", "/services/{name}")
    def one(req):
        require(req.params["name"] in targets, 404, "unknown service")
        return 200, summarize(req.params["name"])

    @svc.app.route("GET", "/trace/{rid}")
    def trace(req):
        with lock:
            recs = list(spans.get(req.params["rid"], []))
        require(recs, 404, "trace not found (unknown or expired request id)")
        hops = [
            {
                k: r.get(k)
                for k in ("ts", "svc", "caller", "method", "path", "status", "dur_ms")
                if r.get(k)
            }
            for r in recs
            if r.get("type") == "access"
        ]
        events = [
            {k: v for k, v in r.items() if k not in ("req",)}
            for r in recs
            if r.get("type") != "access" and r.get("level") in ("warn", "error")
        ]
        hops.sort(key=lambda h: h.get("ts", ""))
        return 200, {"request_id": req.params["rid"], "hops": hops, "events": events[:20]}

    @svc.app.route("GET", "/traces")
    def traces(req):
        only_errors = req.arg("errors") == "1"
        out = []
        with lock:
            for rid, recs in reversed(spans.items()):
                st = [
                    int(r["status"])
                    for r in recs
                    if r.get("type") == "access" and r.get("status", "").isdigit()
                ]
                if only_errors and not any(s >= 500 for s in st):
                    continue
                edge = next(
                    (r for r in recs if r.get("svc") == "edge" and r.get("type") == "access"), None
                )
                out.append(
                    {
                        "request_id": rid,
                        "path": edge and edge.get("path"),
                        "status": max(st) if st else None,
                    }
                )
                if len(out) >= int(req.arg("limit", 10)):
                    break
        return 200, {"traces": out}

    svc.run()


if __name__ == "__main__":
    main()
