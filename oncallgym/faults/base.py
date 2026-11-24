import os
import random
import time
from dataclasses import dataclass

import yaml

CATEGORIES = (
    "config",
    "resilience_config",
    "resource_exhaustion",
    "auth",
    "bad_deploy",
    "schema",
    "performance",
    "data",
    "dependency_down",
)
TIERS = {"easy": 12, "medium": 14, "hard": 16}


@dataclass(frozen=True)
class GroundTruth:
    category: str
    component: str
    description: str
    also_accepted: tuple[str, ...] = ()
    also_accepted_categories: tuple[str, ...] = ()

    def category_ok(self, answer: str) -> bool:
        a = (answer or "").strip().lower()
        return a == self.category or a in self.also_accepted_categories

    def component_ok(self, answer: str) -> bool:
        a = (answer or "").strip().lower()
        return a == self.component or a in self.also_accepted


class Fault:
    family: str = ""
    spec_number: int = 0
    category: str = ""
    tier: str = "easy"
    restart_fixable: bool = False
    breaks: tuple[str, ...] = ("journeys",)
    settle_s: float = 3.0

    def __init__(self, variant: int = 0):
        self.variant = variant
        self.rng = random.Random(f"{self.family}:{variant}")
        self.params = self.choose_params(self.rng)

    def choose_params(self, rng: random.Random) -> dict:
        return {}

    @property
    def component(self) -> str:
        raise NotImplementedError

    also_accepted: tuple[str, ...] = ()
    also_accepted_categories: tuple[str, ...] = ()

    @property
    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            self.category,
            self.component,
            self.describe(),
            self.also_accepted,
            self.also_accepted_categories,
        )

    def root_cause_fixed(self, stack) -> bool:
        raise NotImplementedError

    @property
    def turn_limit(self) -> int:
        return TIERS[self.tier]

    def describe(self) -> str:
        raise NotImplementedError

    def alert(self) -> str:
        raise NotImplementedError

    def configure(self, stack):
        pass

    def inject(self, stack):
        pass

    def oracle(self, stack):
        raise NotImplementedError

    def cheats(self) -> dict:
        return {}

    @property
    def variant_id(self) -> str:
        return f"{self.family}/{self.variant}"


def config_path(stack, name: str) -> str:
    return os.path.join(stack.home, "etc", name if name.endswith(".yaml") else f"{name}.yaml")


def effective_config_path(stack, service: str) -> str:
    from northwind.lib.service import resolve_config_file
    from northwind.supervisor import read_env_file, safe_env

    env = safe_env(read_env_file(os.path.join(stack.home, "etc", "env", f"{service}.env")))
    return resolve_config_file(stack.home, service, env.get("NW_CONFIG_FILE"))


def effective_config(stack, service: str) -> dict:
    return read_yaml(effective_config_path(stack, service))


def effective_flags(stack) -> dict:
    rel = effective_config(stack, "flags").get("flags_file", "etc/feature-flags.yaml")
    return read_yaml(os.path.join(stack.home, rel))


def read_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def edit_yaml(path: str, fn):
    with open(path) as f:
        text = f.read()
    header = "".join(line for line in text.splitlines(keepends=True) if line.startswith("#"))
    doc = yaml.safe_load(text) or {}
    fn(doc)
    with open(path, "w") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)


def edit_service_config(stack, service: str, fn):
    edit_yaml(config_path(stack, service), fn)


def configure_quietly(stack, fault):
    times = _snapshot_times(stack.home)
    ref = max(times.values(), key=lambda t: t[1], default=None)
    fault.configure(stack)
    paths = list(_snapshot_times(stack.home))
    for p in sorted(paths, key=lambda q: (os.path.isdir(q), -q.count(os.sep))):
        t = times.get(p, ref)
        if t:
            os.utime(p, t)


def _snapshot_times(home: str) -> dict:
    out = {}
    for top in ("etc", "data"):
        base = os.path.join(home, top)
        for root, dirs, files in os.walk(base):
            for name in [*files, *dirs, ""]:
                p = os.path.join(root, name) if name else root
                try:
                    st = os.stat(p)
                except FileNotFoundError:
                    continue
                out[p] = (st.st_atime, st.st_mtime)
    return out


def start_with_fault(stack, fault, settle: bool = True):
    if not stack.ports:
        stack.create()
    configure_quietly(stack, fault)
    stack.start(timeout_s=45, require_ready=False)
    time.sleep(2)
    fault.inject(stack)
    if settle:
        time.sleep(fault.settle_s)
    return stack


def restart_and_wait(stack, service: str, timeout_s: float = 20.0) -> bool:
    before = {s["name"]: s for s in stack.services()}[service]["pid"]
    stack.restart(service)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = {s["name"]: s for s in stack.services()}[service]
        if st["ready"] and st["pid"] and st["pid"] != before:
            return True
        time.sleep(0.2)
    return False
