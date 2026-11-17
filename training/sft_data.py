"""Build SFT warm-start data from scripted oracle trajectories (our own data only; no
model outputs). Each fault family provides solution_steps(): a short investigation, the
fix and the submission, expressed as tool calls with a one-line rationale. The script
plays them through a real Episode (same tools, limits and verifier as training and eval),
keeps only episodes the verifier scores as solved, and writes chat-format rows:

    python training/sft_data.py --split train --variants 0-7 --out data/sft_train.jsonl

Row: {"messages": [system, user, assistant(tool_calls), tool, ..., assistant(final)],
      "tools": <schemas>, "variant": ..., "reward": ...}
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oncallgym.episode import Episode, tool_schemas  # noqa: E402
from oncallgym.tasks import task_list  # noqa: E402


def play(family: str, variant: int) -> dict | None:
    ep = Episode(family, variant)
    try:
        messages = ep.start()
        steps = ep.fault.solution_steps(ep.stack)
        for i, (tool, args, thought) in enumerate(steps):
            out = ep.call_tool(tool, args)
            call = {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }
            messages.append({"role": "assistant", "content": thought, "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": f"call_{i}", "content": out})
            if ep.done:
                break
        b = ep.finish()
        if not b.solved:
            print(f"skip {family}/{variant}: not solved {b.to_dict()['details']}", flush=True)
            return None
        messages.append({"role": "assistant", "content": "Fixed and submitted."})
        return {
            "messages": messages,
            "tools": tool_schemas(),
            "variant": f"{family}/{variant}",
            "reward": b.reward,
            "turns": ep.turns,
        }
    finally:
        ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--variants", default="0-3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.variants.split("-"))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    import concurrent.futures as cf

    jobs = [(fam, v) for fam, v in task_list(a.split, variants=hi + 1) if v >= lo]
    n = 0
    with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        for row in ex.map(lambda j: play(*j), jobs):
            if row:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
    print(f"wrote {n} rows to {a.out}")


if __name__ == "__main__":
    main()
