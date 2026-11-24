"""Build SFT warm-start data from scripted oracle trajectories (our own scripts only; no
model outputs), played through real episodes and tokenized exactly like GRPO rollouts
(oncallgym.runner.TokenRenderer: same chat template, tool schemas, enable_thinking=False,
Hermes <tool_call> format). Each row stores input_ids and a mask that is 1 only on
assistant tokens, so the loss is on assistant tokens only.

Variety (all scripted): random extra investigation steps before the fix, harmless red
herrings that are looked at and dismissed, a wrong first guess that gets corrected, the
order of read-only investigation steps shuffled, and several phrasings per thought.
Only episodes the verifier scores as solved are kept.

    python training/sft_data.py --split train --variants 0-7 --per-task 4 \\
        --tokenizer Qwen/Qwen3-1.7B --out data/sft_train.jsonl
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oncallgym.episode import Episode  # noqa: E402
from oncallgym.loop import AgentLoop  # noqa: E402
from oncallgym.runner import TokenRenderer  # noqa: E402
from oncallgym.tasks import task_list  # noqa: E402

MUTATING = {"restart", "submit_diagnosis"}
OPENERS = [
    (
        "list_services",
        {},
        [
            "Start with an overview of service health.",
            "First, which services are up and ready?",
            "Check the state of every service.",
        ],
    ),
    (
        "metrics",
        {},
        [
            "Look at error rates and latency across services.",
            "Which service shows errors or latency?",
            "Get the metrics overview first.",
        ],
    ),
    (
        "topology",
        {},
        [
            "Recall how the services depend on each other.",
            "Check the dependency map before digging in.",
        ],
    ),
]
# Red herrings and wrong guesses teach "this looked fine, it was not the cause". They only
# use services that are never a held-out (test split) root cause, so the data does not teach
# the model to dismiss auth, cache, payments, ... (checked in tests/test_sft_data.py), and
# never the current episode's own root-cause service.
HERRING_SERVICES = ["scheduler", "inventory", "catalog", "flags"]
HERRING_THOUGHTS = [
    "Those warnings look like routine noise, not the incident. Moving on.",
    "Nothing there explains the symptom; that is normal background logging.",
    "Harmless warnings only. The cause is elsewhere.",
]
WRONG_GUESSES = {
    "scheduler": "cat etc/scheduler.yaml",
    "inventory": "cat etc/inventory.yaml",
    "edge": "grep -n timeout_ms etc/edge.yaml",
    "flags": "cat etc/env/flags.env",
}
WRONG_THOUGHTS = [
    "Let me rule out a config problem here first.",
    "Maybe this config is involved; check it.",
]
CORRECTIONS = [
    "That looks fine, so this is not it. Back to the actual symptom.",
    "Nothing wrong there. Focus on what the logs pointed at.",
]


def is_mutating(step):
    tool, args, _ = step
    if tool in MUTATING:
        return True
    cmd = args.get("command", "") if tool == "exec" else ""
    return any(w in cmd for w in ("sed -i", "-X POST", ">>", " > "))


def vary(steps, rng, component):
    """Scripted variations around a family's solution_steps."""
    steps = list(steps)
    first_mut = next((i for i, s in enumerate(steps) if is_mutating(s)), len(steps))
    investigate, rest = steps[:first_mut], steps[first_mut:]
    if len(investigate) > 1 and rng.random() < 0.4:
        a, b = rng.sample(range(len(investigate)), 2)
        investigate[a], investigate[b] = investigate[b], investigate[a]
    extra = []
    for tool, args, thoughts in rng.sample(OPENERS, rng.randint(0, 2)):
        extra.append((tool, args, rng.choice(thoughts)))
    if rng.random() < 0.35:
        svc = rng.choice([x for x in HERRING_SERVICES if x != component])
        extra.append(
            ("logs", {"service": svc, "tail": 5, "grep": "warn"}, f"Check {svc} for warnings too.")
        )
        extra.append(("__note__", {}, rng.choice(HERRING_THOUGHTS)))
    if rng.random() < 0.3:
        cmd = rng.choice([c for svc, c in WRONG_GUESSES.items() if svc != component])
        extra.append(("exec", {"command": cmd}, rng.choice(WRONG_THOUGHTS)))
        extra.append(("__note__", {}, rng.choice(CORRECTIONS)))
    return extra + investigate + rest


def assistant_text(thought, tool, args):
    call = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
    return f"{thought}\n<tool_call>\n{call}\n</tool_call>"


def play(family, variant, aug, renderer, eos_ids):
    rng = random.Random(f"sft:{family}:{variant}:{aug}")
    ep = Episode(family, variant)
    loop = AgentLoop(ep)
    try:
        loop.start()
        steps = ep.fault.solution_steps(ep.stack)
        if aug > 0:
            steps = vary(steps, rng, ep.fault.ground_truth.component)
        input_ids = list(renderer.prompt_ids(loop.messages))
        mask = [0] * len(input_ids)
        pending_note = None
        for tool, args, thought in steps:
            if tool == "__note__":
                pending_note = thought
                continue
            if pending_note:
                thought = f"{pending_note} {thought}"
                pending_note = None
            text = assistant_text(thought, tool, args)
            ids = renderer.tok.encode(text, add_special_tokens=False) + eos_ids
            input_ids += ids
            mask += [1] * len(ids)
            env = loop.step(text)
            if loop.done:
                break
            suffix = renderer.suffix_ids(env)
            input_ids += suffix
            mask += [0] * len(suffix)
        b = ep.finish()
        if not b.solved:
            print(f"skip {family}/{variant}/{aug}: not solved", flush=True)
            return None
        return {
            "input_ids": input_ids,
            "assistant_mask": mask,
            "variant": f"{family}/{variant}",
            "aug": aug,
            "turns": ep.turns,
            "reward": b.reward,
            "n_tokens": len(input_ids),
        }
    finally:
        ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--variants", default="0-7")
    ap.add_argument(
        "--per-task", type=int, default=4, help="trajectories per task (1 plain + n-1 varied)"
    )
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    import concurrent.futures as cf

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    renderer = TokenRenderer(tok)
    eos_ids = [tok.convert_tokens_to_ids("<|im_end|>")]
    lo, hi = (int(x) for x in a.variants.split("-"))
    jobs = [
        (f, v, k)
        for f, v in task_list(a.split, variants=hi + 1)
        if v >= lo
        for k in range(a.per_task)
    ]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    n = 0
    with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(lambda j: play(*j, renderer, eos_ids), jobs):
            if row:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
    print(f"wrote {n}/{len(jobs)} rows to {a.out}")


if __name__ == "__main__":
    main()
