"""Training (TRL) and eval (verifiers) adapters expose exactly the core tools and prompt."""

import asyncio
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environments" / "oncallgym_env" / "src"))

from oncallgym.episode import TOOL_NAMES, Tools, tool_schemas  # noqa: E402
from oncallgym.prompt import system_prompt, task_prompt  # noqa: E402
from training.trl_env import OnCallEnv  # noqa: E402


def test_trl_methods_match_core_tools():
    public = sorted(
        n
        for n, _ in inspect.getmembers(OnCallEnv, inspect.isfunction)
        if not n.startswith("_") and n not in ("reset", "get_reward", "close")
    )
    assert public == sorted(TOOL_NAMES)
    for n in TOOL_NAMES:
        assert inspect.signature(getattr(OnCallEnv, n)) == inspect.signature(getattr(Tools, n))
        assert getattr(OnCallEnv, n).__doc__ == getattr(Tools, n).__doc__


def test_trl_delegate_forwards_args():
    calls = []

    class FakeEpisode:
        done = False

        def call_tool(self, name, args):
            calls.append((name, args))
            return "ok"

    env = OnCallEnv()
    env.episode = FakeEpisode()
    assert env.logs(service="orders", grep="error") == "ok"
    assert env.logs(nonsense=1) == "ok"
    assert calls == [("logs", {"service": "orders", "grep": "error"}), ("logs", {"nonsense": 1})]


def test_verifiers_toolset_serves_core_schemas():
    from oncallgym_env.toolset import OnCallToolset
    from verifiers.v1.mcp import SharedToolsetConfig

    tools = asyncio.run(OnCallToolset(SharedToolsetConfig()).list_tools())
    served = [
        {"name": t.name, "description": t.description, "parameters": t.input_schema} for t in tools
    ]
    assert served == [s["function"] for s in tool_schemas()]


def test_verifiers_task_prompt_is_core_prompt():
    from oncallgym_env.taskset import OnCallConfig, OnCallTaskset

    from oncallgym.faults import make_fault

    ts = OnCallTaskset(OnCallConfig(variants=1))
    tasks = list(ts.load())
    assert tasks
    for t in tasks:
        f = make_fault(t.data.family, t.data.variant)
        assert t.data.system_prompt == system_prompt()
        assert t.data.prompt == task_prompt(f.alert(), f.turn_limit)


def test_trl_dataset_prompt_is_core_prompt():
    from oncallgym.faults import make_fault
    from training.dataset import build_rows

    rows = build_rows("train", variants=range(2))
    assert rows
    for r in rows:
        f = make_fault(r["family"], r["variant"])
        assert r["prompt"][0]["content"] == system_prompt()
        assert r["prompt"][1]["content"] == task_prompt(f.alert(), f.turn_limit)


def test_trl_rendered_schemas_match_core():
    """TRL renders tool schemas with transformers' parser; they must equal ours."""
    import pytest

    tu = pytest.importorskip("transformers.utils")
    for s in tool_schemas():
        name = s["function"]["name"]
        trl = tu.get_json_schema(getattr(OnCallEnv, name))
        assert trl["function"]["name"] == name
        assert (
            trl["function"]["parameters"].get("required", [])
            == (s["function"]["parameters"]["required"])
        )
        assert set(trl["function"]["parameters"]["properties"]) == set(
            s["function"]["parameters"]["properties"]
        )
