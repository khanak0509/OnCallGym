import asyncio
import json
import os
import sys
import uuid
from collections.abc import Iterator

import httpx
from pydantic import Field
from verifiers.v1.mcp import SharedToolsetConfig, Toolset
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import Task, TaskConfig, TaskData
from verifiers.v1.taskset import Taskset, TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.utils.decorators import metric, reward, stop

from oncallgym.faults import FAMILIES, make_fault
from oncallgym.prompt import system_prompt, task_prompt
from oncallgym.tasks import task_list

from .toolset import OnCallState, OnCallToolset


class OnCallTaskConfig(TaskConfig):
    ready_timeout_s: float = 90.0
    trajectory_dir: str = ""
    """If set, each episode's full trajectory (tool calls, outputs, verifier breakdown) is
    written there as <task name>-<trace id>.json (the runtime workdir is deleted)."""


class OnCallConfig(TasksetConfig):
    split: str = "all"
    """train, test or all (split by fault family)."""
    families: list[str] = Field(default_factory=list)
    """Restrict to these families (default: all in the split)."""
    variants: int = 3
    tools: SharedToolsetConfig = SharedToolsetConfig()
    task: OnCallTaskConfig = OnCallTaskConfig()


class OnCallData(TaskData):
    family: str
    variant: int
    seed: int


class OnCallTask(Task[OnCallData, OnCallState, OnCallTaskConfig]):
    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        d = self.data
        traj = "trajectory.json"
        if self.config.trajectory_dir:
            os.makedirs(self.config.trajectory_dir, mode=0o700, exist_ok=True)
            name = d.name.replace("/", "_")
            traj = os.path.join(self.config.trajectory_dir, f"{name}-{uuid.uuid4().hex[:8]}.json")
        task = json.dumps({"family": d.family, "variant": d.variant, "seed": d.seed})
        await runtime.write("task.json", task.encode())
        argv = [
            sys.executable,
            "-m",
            "oncallgym.episode_host",
            "--task-file",
            "task.json",
            "--ready-file",
            "episode.json",
            "--trajectory-file",
            traj,
        ]
        await runtime.run_background(argv, {}, "episode_host.log")
        deadline = asyncio.get_running_loop().time() + self.config.ready_timeout_s
        while True:
            try:
                info = json.loads(await runtime.read("episode.json"))
                break
            except (FileNotFoundError, ValueError):
                if asyncio.get_running_loop().time() > deadline:
                    log = (await runtime.read("episode_host.log")).decode(errors="replace")
                    raise RuntimeError(f"episode did not start:\n{log[-2000:]}") from None
                await asyncio.sleep(0.5)
        if "error" in info:
            raise RuntimeError(f"episode failed to start: {info['error']}")
        trace.state.episode_url = info["url"]
        trace.state.episode_token = info["token"]

    @stop
    def submitted(self, trace: Trace) -> bool:
        return trace.state.done

    @reward(weight=1.0)
    async def oncall(self, trace: Trace) -> float:
        async with httpx.AsyncClient(timeout=180) as client:
            b = (
                await client.post(
                    f"{trace.state.episode_url}/finish",
                    headers={"X-Episode-Token": trace.state.episode_token},
                )
            ).json()
        trace.info["oncallgym"] = b
        for key in ("R", "P", "C", "G"):
            trace.record_metric(key, float(b[key]))
        trace.record_metric("solved", float(b["solved"]))
        trace.record_metric(f"termination_{b['termination']}", 1.0)
        return float(b["reward"])

    @metric
    async def turns(self, trace: Trace) -> float:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                f"{trace.state.episode_url}/state",
                headers={"X-Episode-Token": trace.state.episode_token},
            )
            return float(r.json()["turns"])


class OnCallTaskset(Taskset[OnCallTask, OnCallConfig]):
    @classmethod
    def toolsets(cls, config: OnCallConfig) -> list[Toolset]:
        return [OnCallToolset(config.tools)]

    def load(self) -> Iterator[OnCallTask]:
        for family, variant in task_list(
            self.config.split, self.config.families, self.config.variants
        ):
            f = make_fault(family, variant)
            yield OnCallTask(
                OnCallData(
                    name=f.variant_id,
                    system_prompt=system_prompt(),
                    prompt=task_prompt(f.alert(), f.turn_limit),
                    family=family,
                    variant=variant,
                    seed=variant,
                ),
                self.config.task,
            )


assert FAMILIES
