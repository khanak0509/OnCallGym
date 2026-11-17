"""TRL rollout_func: real OnCallGym episodes through the shared runner (oncallgym.runner),
which is also what eval uses. Generation goes through TRL's vLLM client (weights already
synced by TRL before this is called); environment tokens are masked via env_mask.

The prompt rows only carry chat messages, so each prompt is mapped back to its task by
its text (variants with identical alerts are interchangeable here; one is picked at
random each time).
"""

import json
import os
import random
import threading

from oncallgym.runner import run_episodes

_LOG: list[dict] = []
_LOG_LOCK = threading.Lock()


def drain_episode_log() -> list[dict]:
    with _LOG_LOCK:
        out = list(_LOG)
        _LOG.clear()
    return out


def make_rollout_func(
    rows, rollouts_dir=None, max_turn_tokens=768, max_model_len=8192, run=run_episodes
):
    by_prompt: dict[str, list[dict]] = {}
    for r in rows:
        by_prompt.setdefault(json.dumps(r["prompt"], sort_keys=True), []).append(r)
    rng = random.Random(0)
    counter = [0]

    def rollout_func(prompts, trainer):
        # TRL's RepeatSampler already repeats each prompt num_generations times in a row, so
        # one episode per received prompt; a group must all play the same task.
        n = trainer.num_generations
        vg = trainer.vllm_generation
        tok = trainer.processing_class
        specs = []
        for i, p in enumerate(prompts):
            if i % n == 0:
                row = rng.choice(by_prompt[json.dumps(p, sort_keys=True)])
            specs.append({"family": row["family"], "variant": row["variant"], "seed": row["seed"]})

        def generate(batch, max_new):
            saved = vg.max_completion_length
            vg.max_completion_length = max_new
            try:
                _, ids, lps, _, _ = vg.generate(prompts=batch, images=None, num_generations=1)
            finally:
                vg.max_completion_length = saved
            return [(i, [lp[0] for lp in lp_seq]) for i, lp_seq in zip(ids, lps, strict=True)]

        results = run(
            specs, generate, tok, max_turn_tokens=max_turn_tokens, max_model_len=max_model_len
        )
        for r in results:
            b = r["breakdown"]
            with _LOG_LOCK:
                _LOG.append(
                    {
                        "variant": r["variant"],
                        "reward": b["reward"],
                        "R": b["R"],
                        "P": b["P"],
                        "C": b["C"],
                        "G": b["G"],
                        "termination": b["termination"],
                        "turns": b["details"].get("turns"),
                        "tokens": len(r["prompt_ids"]) + len(r["completion_ids"]),
                    }
                )
            if rollouts_dir:
                os.makedirs(rollouts_dir, mode=0o700, exist_ok=True)
                counter[0] += 1
                path = os.path.join(
                    rollouts_dir, f"{counter[0]:05d}-{r['variant'].replace('/', '_')}.json"
                )
                with open(path, "w") as f:
                    json.dump({"breakdown": b, "trajectory": r["trajectory"]}, f, default=str)
        return {
            "prompt_ids": [r["prompt_ids"] for r in results],
            "completion_ids": [r["completion_ids"] for r in results],
            "logprobs": [r["logprobs"] for r in results],
            "env_mask": [r["env_mask"] for r in results],
            "oncall_reward": [r["breakdown"]["reward"] for r in results],
        }

    return rollout_func


def oncall_reward(completions=None, oncall_reward=None, **kwargs):
    return list(oncall_reward)
