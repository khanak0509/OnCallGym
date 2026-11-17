"""Fault framework.

A fault family describes one kind of production incident. A variant (family + variant
number) fixes the random parameters, so episodes are reproducible.

    f = make_fault("edge_wrong_upstream_port", variant=3)
    f.inject(stack)        # after the stack is up and healthy
    f.oracle(stack)        # scripted fix, using only what the agent can do
    f.ground_truth         # hidden from the agent

Injection and oracle fixes only edit files inside the episode directory and restart
services through the supervisor, the same operations the agent has.
"""

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
TIERS = {"easy": 8, "medium": 12, "hard": 16}


@dataclass(frozen=True)
class GroundTruth:
    category: str
    component: str
    description: str


class Fault:
    family: str = ""
    spec_number: int = 0
    category: str = ""
    tier: str = "easy"
    restart_fixable: bool = False
    # which probe parts are expected to fail after injection (journeys, window, invariants)
    breaks: tuple[str, ...] = ("journeys",)
    settle_s: float = 3.0  # wait after injection before symptoms are expected

    def __init__(self, variant: int = 0):
        self.variant = variant
        self.rng = random.Random(f"{self.family}:{variant}")
        self.params = self.choose_params(self.rng)

    def choose_params(self, rng: random.Random) -> dict:
        return {}

    @property
    def component(self) -> str:
        raise NotImplementedError

    @property
    def ground_truth(self) -> GroundTruth:
        return GroundTruth(self.category, self.component, self.describe())

    @property
    def turn_limit(self) -> int:
        return TIERS[self.tier]

    def describe(self) -> str:
        raise NotImplementedError

    def alert(self) -> str:
        """What the pager says: symptoms only, never the cause."""
        raise NotImplementedError

    def inject(self, stack):
        raise NotImplementedError

    def oracle(self, stack):
        raise NotImplementedError

    def cheats(self) -> dict:
        """Family-specific cheat attempts: name -> fn(stack). Generic cheats live in
        oncallgym/cheats.py."""
        return {}

    @property
    def variant_id(self) -> str:
        return f"{self.family}/{self.variant}"


# helpers shared by faults and oracles


def config_path(stack, name: str) -> str:
    return os.path.join(stack.home, "etc", name if name.endswith(".yaml") else f"{name}.yaml")


def read_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def edit_yaml(path: str, fn):
    """Load, mutate with fn(doc), write back (keeping the header comment)."""
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
