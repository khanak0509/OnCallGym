"""Training (TRL) and eval (verifiers) adapters expose exactly the core tools and prompt."""

import asyncio
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environments" / "oncallgym_env" / "src"))

from oncallgym.episode import tool_schemas  # noqa: E402
from oncallgym.prompt import system_prompt, task_prompt  # noqa: E402


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


def test_training_and_eval_share_the_runner():
    """Training (TRL rollout_func) and eval drive episodes through the same function, which
    uses the same loop (nudge, parsing) and renderer (template, schemas, no thinking)."""

    import oncallgym.evaluate as ev
    import training.rollout as ro
    from oncallgym import runner

    assert ro.run_episodes is runner.run_episodes
    assert ev.run_episodes is runner.run_episodes
    src = inspect.getsource(runner)
    assert "AgentLoop" in src and "TokenRenderer" in src
    assert runner.TEMPLATE_KWARGS == {"enable_thinking": False}


def test_sft_data_uses_the_runner_renderer():
    import training.sft_data as sd
    from oncallgym import runner

    assert sd.TokenRenderer is runner.TokenRenderer
    assert sd.AgentLoop.__module__ == "oncallgym.loop"
