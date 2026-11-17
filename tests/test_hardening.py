"""Regression tests for the checkpoint 3 sandbox hardening (HACKS.md 23-25)."""

import os
import platform

import pytest
import yaml

from northwind.lib.service import resolve_config_file
from oncallgym.faults import make_fault
from oncallgym.faults.base import _snapshot_times, configure_quietly, effective_config
from oncallgym.stack import Stack

linux_only = pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")


def _home(tmp_path):
    for d in ("etc/env", "etc/archive"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "etc/scheduler.yaml").write_text(yaml.safe_dump({"log": {"level": "info"}}))
    (tmp_path / "etc/archive/scheduler-old.yaml").write_text(
        yaml.safe_dump({"log": {"level": "warn"}})
    )
    (tmp_path / "etc/s2.yaml").write_text(yaml.safe_dump({"log": {"level": "error"}}))
    return tmp_path


def test_config_override_only_to_archive(tmp_path):
    h = str(_home(tmp_path))
    assert resolve_config_file(h, "scheduler", None).endswith("etc/scheduler.yaml")
    assert resolve_config_file(h, "scheduler", "etc/archive/scheduler-old.yaml").endswith(
        "etc/archive/scheduler-old.yaml"
    )
    for bad in ("etc/s2.yaml", "etc/archive/../s2.yaml", "/etc/passwd", "data/x.yaml"):
        with pytest.raises(SystemExit):
            resolve_config_file(h, "scheduler", bad)


def test_verifier_reads_effective_config(tmp_path):
    h = _home(tmp_path)
    stack = type("S", (), {"home": str(h)})()
    (h / "etc/env/scheduler.env").write_text("NW_CONFIG_FILE=etc/archive/scheduler-old.yaml\n")
    assert effective_config(stack, "scheduler")["log"]["level"] == "warn"


@pytest.mark.stack
@linux_only
@pytest.mark.parametrize(
    "family", ["jwt_key_rotation_mismatch", "missing_migration", "orders_config_drift"]
)
def test_fault_injection_leaves_no_timestamps(tmp_path, family):
    s = Stack(seed=3, root=str(tmp_path))
    try:
        s.create()
        before = _snapshot_times(s.home)
        configure_quietly(s, make_fault(family, 0))
        after = _snapshot_times(s.home)
        newest_before = max(t[1] for t in before.values())
        # modification times (access times change whenever anyone lists a directory)
        changed = [p for p, t in after.items() if p in before and t[1] != before[p][1]]
        newer = [p for p, t in after.items() if t[1] > newest_before]
        assert not changed and not newer, (changed, newer)
    finally:
        s.cleanup()


@pytest.mark.stack
@linux_only
def test_agent_cannot_read_code_or_add_archived_config(tmp_path):
    import northwind
    from oncallgym.episode import Episode

    root = os.path.dirname(os.path.dirname(os.path.abspath(northwind.__file__)))
    ep = Episode("catalog_bad_build_flag", 0, 0, root=str(tmp_path))
    try:
        ep.start()
        out = ep.call_tool(
            "exec",
            {
                "command": f"ls {root}/oncallgym; cat {root}/oncallgym/"
                "faults/families.py | head -1; echo rc=$?"
            },
        )
        assert "Permission denied" in out and "class " not in out
        ep.call_tool("exec", {"command": "cp etc/catalog.yaml etc/archive/catalog-x.yaml"})
        b = ep.finish()
        assert b.G == 0.0 and "before_restart:no_new_archived_configs" in b.details["integrity"]
    finally:
        ep.close()
