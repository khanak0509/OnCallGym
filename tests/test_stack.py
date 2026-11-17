"""Integration tests that start real stacks. Linux only (dev container or Kaggle)."""

import os
import platform
import signal
import time

import pytest

from oncallgym import stack as stack_mod
from oncallgym.probes import Probe
from oncallgym.stack import HostNotAllowed, Stack, leftover_processes

linux_only = pytest.mark.skipif(
    platform.system() != "Linux", reason="stacks only run inside the container / Linux VMs"
)


def test_refuses_macos_host(monkeypatch, tmp_path):
    monkeypatch.setattr(stack_mod.platform, "system", lambda: "Darwin")
    s = Stack(root=str(tmp_path))
    with pytest.raises(HostNotAllowed):
        s.start()
    s.cleanup()


@pytest.fixture(scope="module")
def running(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("nw"))
    s = Stack(seed=5, root=root)
    s.create().start()
    time.sleep(3)
    yield s
    s.cleanup()
    assert leftover_processes(root) == []
    assert not os.path.exists(s.home)


def wait_ready(s, name, old_pid, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = {x["name"]: x for x in s.services()}[name]
        if st["ready"] and st["pid"] and st["pid"] != old_pid:
            return st
        time.sleep(0.3)
    raise AssertionError(f"{name} not ready again: {st}")


@pytest.mark.stack
@linux_only
def test_probe_passes(running):
    r = Probe(running).run(journeys=2, window_s=4)
    assert r["passed"], r


@pytest.mark.stack
@linux_only
def test_restart_tool_path(running):
    before = {x["name"]: x for x in running.services()}["catalog"]
    running.restart("catalog")
    after = wait_ready(running, "catalog", before["pid"])
    assert after["pid"] != before["pid"] and after["restarts"] == before["restarts"] + 1


@pytest.mark.stack
@linux_only
def test_crashed_service_is_restarted(running):
    before = {x["name"]: x for x in running.services()}["notify"]
    os.kill(before["pid"], signal.SIGKILL)
    after = wait_ready(running, "notify", before["pid"])
    assert after["last_exit"] == -9


@pytest.mark.stack
@linux_only
def test_stacks_do_not_collide(tmp_path):
    root = str(tmp_path)
    a, b = Stack(seed=1, root=root), Stack(seed=2, root=root)
    try:
        a.create()
        b.create()
        assert a.block != b.block
        assert not set(a.ports.values()) & set(b.ports.values())
        a.start()
        b.start()
        ra, rb = Probe(a).journey(), Probe(b).journey()
        assert ra["ok"] and rb["ok"]
    finally:
        a.cleanup()
        b.cleanup()
    assert leftover_processes(root) == []


@pytest.mark.stack
@linux_only
def test_stop_kills_everything_even_if_supervisor_is_killed(tmp_path):
    s = Stack(seed=3, root=str(tmp_path))
    s.create().start()
    pids = [x["pid"] for x in s.services()]
    os.kill(s.proc.pid, signal.SIGKILL)
    s.cleanup()
    time.sleep(0.5)
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
