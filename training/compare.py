"""Paired comparison of two eval runs on the same tasks (oncallgym.evaluate output dirs).

    python training/compare.py runs/post_sft runs/post_grpo

Per task: solve rate and mean reward of A and B. Overall difference with a bootstrap CI over
tasks (tasks are the unit, samples within a task are correlated) and a sign-flip permutation
p-value on the per-task differences.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


def load(d):
    by_task = defaultdict(list)
    for f in sorted(Path(d, "trajectories").glob("*.json")):
        task = re.sub(r"-\d+$", "", f.stem)
        ev = json.loads(f.read_text())
        fin = next((e for e in reversed(ev) if e.get("kind") == "finish"), None)
        b = (fin or {}).get("breakdown") or {}
        by_task[task].append((float(bool(b.get("solved"))), float(b.get("reward", 0.0))))
    return by_task


def stats(diffs, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    mean = sum(diffs) / n
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(iters))
    ci = (boots[int(0.025 * iters)], boots[int(0.975 * iters)])
    hits = sum(
        abs(sum(x if rng.random() < 0.5 else -x for x in diffs) / n) >= abs(mean) - 1e-12
        for _ in range(iters)
    )
    return mean, ci, hits / iters


def main(a, b):
    A, B = load(a), load(b)
    tasks = sorted(set(A) & set(B))
    head = ["nA", "nB", "solveA", "solveB", "rewA", "rewB"]
    print(f"{'task':40s} " + " ".join(f"{h:>7s}" for h in head))
    ds, dr = [], []
    for t in tasks:
        sa = sum(x for x, _ in A[t]) / len(A[t])
        sb = sum(x for x, _ in B[t]) / len(B[t])
        ra = sum(r for _, r in A[t]) / len(A[t])
        rb = sum(r for _, r in B[t]) / len(B[t])
        ds.append(sb - sa)
        dr.append(rb - ra)
        print(f"{t:40s} {len(A[t]):3d} {len(B[t]):3d} {sa:7.3f} {sb:7.3f} {ra:6.3f} {rb:6.3f}")
    nA = sum(len(A[t]) for t in tasks)
    nB = sum(len(B[t]) for t in tasks)
    sA = sum(x for t in tasks for x, _ in A[t]) / nA
    sB = sum(x for t in tasks for x, _ in B[t]) / nB
    print(f"\n{len(tasks)} tasks; A solve {sA:.4f} ({nA} eps), B solve {sB:.4f} ({nB} eps)")
    for name, d in [("solve rate", ds), ("mean reward", dr)]:
        m, ci, p = stats(d)
        print(
            f"B - A {name}: {m:+.4f}  95% CI over tasks [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
            f"sign-flip p={p:.4f}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
