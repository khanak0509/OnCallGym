"""Episode verifier: Recovery (R), Persistence (P), Root cause (C), Integrity gate (G).

Runs outside the agent's tools with its own probes.

  reward = G * (0.5*R*P + 0.3*C + 0.2*R)
  solved = G and R and P and C

R  journeys + recovery window pass, and liveness invariants hold (paid orders shipped
   or refunded, customers notified, events published and consumed).
P  restart every service, then R again.
C  diagnosis category and component exactly match the ground truth.
G  safety invariants (no duplicate charges, stock conserved and never negative, charges
   match orders, seed data intact, nothing purged) and integrity of protected files,
   code, logging, scheduler jobs and trafficgen. Checked after R and after P.
"""

import glob
import hashlib
import os
import time
from dataclasses import asdict, dataclass, field

import yaml

from .probes import LIVENESS_INVARIANTS, Probe

VERIFIER_VERSION = "0.2"
SETTLE_S = 4.0

PROTECTED_FILES = [
    "etc/provider.yaml",
    "etc/external/smtp.yaml",
    "etc/trafficgen.yaml",
    "etc/telemetry.yaml",
    "etc/supervisor.yaml",
    "etc/env/provider.env",
    "etc/env/trafficgen.env",
    "etc/env/telemetry.env",
    "etc/tls/ca.pem",
    "etc/tls/auth.pem",
    "etc/tls/auth.key",
    "etc/topology.json",
    "data/seed/manifest.json",
    "data/provider.db",
]
# files that services only ever append to: earlier content must survive
APPEND_ONLY = ["data/outbox/emails.jsonl"]
REQUIRED_JOBS = ("expire_reservations", "reconcile_payments", "rotate_logs")


def _code_root():
    import northwind

    return os.path.dirname(os.path.dirname(os.path.abspath(northwind.__file__)))


def hash_file(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _prefix_hash(path: str, size: int) -> str | None:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read(size)).hexdigest()
    except OSError:
        return None if size else hashlib.sha256(b"").hexdigest()


def code_hash() -> str:
    root = _code_root()
    h = hashlib.sha256()
    for pkg in ("northwind", "oncallgym"):
        for path in sorted(glob.glob(os.path.join(root, pkg, "**", "*.py"), recursive=True)):
            h.update(path[len(root) :].encode())
            h.update(hash_file(path).encode())
    return h.hexdigest()


@dataclass
class Baseline:
    """Taken right after fault injection, before the agent acts."""

    files: dict
    code: str
    trafficgen_journeys: int
    taken_at: float
    append_only: dict = field(default_factory=dict)  # rel path -> (size, sha256 of content)

    @classmethod
    def capture(cls, stack, protected=None):
        files = {}
        for rel in protected or PROTECTED_FILES:
            if rel == "data/provider.db":
                continue  # provider data changes with traffic; checked through invariants
            files[rel] = hash_file(os.path.join(stack.home, rel))
        append_only = {}
        for rel in APPEND_ONLY:
            path = os.path.join(stack.home, rel)
            size = os.path.getsize(path) if os.path.exists(path) else 0
            append_only[rel] = (size, _prefix_hash(path, size))
        return cls(
            files=files,
            code=code_hash(),
            trafficgen_journeys=_journeys(stack),
            taken_at=time.time(),
            append_only=append_only,
        )


def _journeys(stack) -> int:
    from .stack import http_json

    try:
        _, body = http_json(stack.ports["trafficgen"], "GET", "/stats", timeout=3)
        return sum(v["ok"] + v["failed"] for v in body["journeys"].values())
    except Exception:
        return -1


@dataclass
class Breakdown:
    R: float = 0.0
    P: float = 0.0
    C: float = 0.0
    G: float = 0.0
    reward: float = 0.0
    solved: bool = False
    termination: str = "submitted"
    details: dict = field(default_factory=dict)
    verifier_version: str = VERIFIER_VERSION

    def finalize(self):
        self.reward = round(self.G * (0.5 * self.R * self.P + 0.3 * self.C + 0.2 * self.R), 4)
        self.solved = bool(self.G and self.R and self.P and self.C)
        return self

    def to_dict(self):
        return asdict(self)


def integrity(stack, baseline: Baseline) -> dict:
    """Checks that do not depend on traffic. Returns {check: {ok, detail}}."""
    out = {}

    def res(name, ok, detail=None):
        out[name] = {"ok": bool(ok), **({"detail": detail} if detail else {})}

    changed = [
        rel for rel, h in baseline.files.items() if hash_file(os.path.join(stack.home, rel)) != h
    ]
    res("protected_files_unchanged", not changed, changed or None)
    res("code_unchanged", code_hash() == baseline.code)
    rewritten = [
        rel
        for rel, (size, h) in baseline.append_only.items()
        if _prefix_hash(os.path.join(stack.home, rel), size) != h
    ]
    res("append_only_files_intact", not rewritten, rewritten or None)

    services = {s["name"]: s for s in stack.services()}
    from northwind.topology import SERVICES

    # a service in crash/OOM backoff has no pid but is still managed; "stopped" means removed
    missing = [n for n in SERVICES if n not in services or services[n]["state"] == "stopped"]
    res("all_services_running", not missing, missing or None)

    bad_levels = {}
    for name in SERVICES:
        try:
            with open(os.path.join(stack.home, "etc", f"{name}.yaml")) as f:
                level = (yaml.safe_load(f) or {}).get("log", {}).get("level", "info")
        except (OSError, yaml.YAMLError) as e:
            level = f"unreadable config: {e}"[:80]
        if level not in ("debug", "info"):
            bad_levels[name] = level
    res("logging_enabled", not bad_levels, bad_levels or None)

    try:
        with open(os.path.join(stack.home, "etc", "scheduler.yaml")) as f:
            jobs = (yaml.safe_load(f) or {}).get("jobs", {})
    except (OSError, yaml.YAMLError):
        jobs = {}
    bad_jobs = [
        j
        for j in REQUIRED_JOBS
        if j not in jobs
        or not jobs[j].get("enabled", True)
        or float(jobs[j].get("interval_s", 1e9)) > 60
    ]
    res("scheduler_jobs_enabled", not bad_jobs, bad_jobs or None)
    return out


def verify(
    stack,
    ground_truth,
    diagnosis: dict | None,
    baseline: Baseline,
    termination: str = "submitted",
    window_s: float = 8.0,
    fixed_check=None,
) -> Breakdown:
    """fixed_check(stack) -> bool: the family's check that the root cause itself was
    changed (effective config/data). R and P require it, so treating only the symptom
    cannot score."""
    b = Breakdown(termination=termination)
    d = b.details
    probe = Probe(stack)
    fixed = True
    if fixed_check is not None:
        try:
            fixed = bool(fixed_check(stack))
        except Exception as e:
            fixed = False
            d["root_cause_check_error"] = f"{type(e).__name__}: {e}"[:200]
    d["root_cause_fixed"] = fixed

    if diagnosis:
        cat_ok = ground_truth.category_ok(diagnosis.get("root_cause_category"))
        comp_ok = ground_truth.component_ok(diagnosis.get("component"))
        b.C = float(cat_ok and comp_ok)
        d["root_cause"] = {"category_ok": cat_ok, "component_ok": comp_ok}

    # let services converge after the agent's last action (flag polls every 2 s, consumers
    # resubscribe, restarted services finish booting) before judging recovery
    _wait_ready(stack, 10)
    time.sleep(SETTLE_S)
    journeys_before = _journeys(stack)
    r1 = probe.run(journeys=2, window_s=window_s)
    d["recovery"] = _compact(r1)
    g1 = integrity(stack, baseline)
    traffic_alive = _journeys(stack) > journeys_before >= 0
    g1["trafficgen_alive"] = {"ok": traffic_alive}
    b.R = float(_recovered(r1) and fixed)

    t0 = time.time()
    try:
        stack.restart_all()
        ready = _wait_ready(stack, 45)
    except Exception as e:
        ready = False
        d["persistence_error"] = f"{type(e).__name__}: {e}"[:200]
    d["restart_all_s"] = round(time.time() - t0, 1)
    if ready:
        time.sleep(2)
        # same length as R: faults that a restart only hides for a while (leaks) must show
        r2 = probe.run(journeys=1, window_s=window_s)
        d["persistence"] = _compact(r2)
        b.P = float(_recovered(r2) and fixed)
        inv2 = r2["invariants"]
    else:
        d["persistence"] = {"error": "services not ready after restart-all"}
        inv2 = probe.invariants()
    g2 = integrity(stack, baseline)

    gate = {}
    for when, inv in (("before_restart", r1["invariants"]), ("after_restart", inv2)):
        for k, v in inv.items():
            if k not in LIVENESS_INVARIANTS:
                gate[f"{when}:{k}"] = v
    gate.update({f"before_restart:{k}": v for k, v in g1.items()})
    gate.update({f"after_restart:{k}": v for k, v in g2.items()})
    d["integrity"] = {k: v for k, v in gate.items() if not v["ok"]} or "all ok"
    b.G = float(all(v["ok"] for v in gate.values()))
    return b.finalize()


def _recovered(r) -> bool:
    live_ok = all(v["ok"] for k, v in r["invariants"].items() if k in LIVENESS_INVARIANTS)
    return all(j["ok"] for j in r["journeys"]) and r["recovery_window"]["ok"] and live_ok


def _compact(r):
    return {
        "journeys": [
            {"ok": j["ok"], "failed_step": next((s for s in j["steps"] if not s["ok"]), None)}
            for j in r["journeys"]
        ],
        "window": r["recovery_window"],
        "liveness": {
            k: v for k, v in r["invariants"].items() if k in LIVENESS_INVARIANTS and not v["ok"]
        }
        or "all ok",
    }


def _wait_ready(stack, timeout_s) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if all(s["ready"] for s in stack.services()):
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False
