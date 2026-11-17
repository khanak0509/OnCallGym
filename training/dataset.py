"""Prompt dataset for GRPO: one row per (family, variant), prompts from oncallgym.prompt
(the same text the eval path sends). TRL passes each row's extra columns to
OnCallEnv.reset(**row).

    rows = build_rows("train", variants=range(8), repeats=1)
"""

from oncallgym.faults import make_fault
from oncallgym.prompt import system_prompt, task_prompt
from oncallgym.tasks import task_list


def build_rows(split: str = "train", variants=range(4), families=None, repeats: int = 1):
    rows = []
    n = max(variants) + 1
    for family, v in task_list(split, families, variants=n):
        if v not in variants:
            continue
        f = make_fault(family, v)
        for _ in range(repeats):
            rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": system_prompt()},
                        {"role": "user", "content": task_prompt(f.alert(), f.turn_limit)},
                    ],
                    "family": family,
                    "variant": v,
                    "seed": v,
                    "tier": f.tier,
                }
            )
    return rows


def to_hf_dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)
