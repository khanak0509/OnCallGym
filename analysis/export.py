import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import ORDER, outcome  # noqa: E402


def export(src, dest, commit, source, model_desc=None):
    meta = json.loads(Path(src, "meta.json").read_text())
    rows = []
    for f in sorted(Path(src, "trajectories").glob("*.json")):
        ev = json.loads(f.read_text())
        fin = next((e for e in reversed(ev) if e.get("kind") == "finish"), None)
        if fin is None:
            continue
        b = fin["breakdown"]
        d = b.get("details") or {}
        rc = d.get("root_cause") or {}
        task, sample = f.stem.rsplit("-", 1)
        rows.append(
            {
                "task": task,
                "sample": int(sample),
                "reward": b["reward"],
                "R": b["R"],
                "P": b["P"],
                "C": b["C"],
                "G": b["G"],
                "solved": bool(b["solved"]),
                "root_cause_fixed": bool(d.get("root_cause_fixed")),
                "category_ok": bool(rc.get("category_ok")),
                "component_ok": bool(rc.get("component_ok")),
                "termination": b["termination"],
                "turns": d.get("turns"),
                "outcome": outcome(b),
            }
        )
    n = len(rows)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["solved"])
    agg = {
        "episodes": n,
        "tasks": len(by_task),
        "solve_rate": sum(r["solved"] for r in rows) / n,
        "pass_at_k": sum(any(v) for v in by_task.values()) / len(by_task),
        "mean_reward": sum(r["reward"] for r in rows) / n,
        **{f"mean_{k}": sum(r[k] for r in rows) / n for k in "RPCG"},
        "mean_turns": sum(r["turns"] or 0 for r in rows) / n,
        "outcomes": {k: sum(r["outcome"] == k for r in rows) / n for k in ORDER},
    }
    out = {
        "provenance": {"commit": commit, "source": source, "model": model_desc or meta["model"]},
        "eval_settings": meta,
        "aggregate": agg,
        "episodes": rows,
    }
    Path(dest).mkdir(parents=True, exist_ok=True)
    Path(dest, "summary.json").write_text(json.dumps(out, indent=1))
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dest")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--model-desc", default=None)
    a = ap.parse_args()
    agg = export(a.src, a.dest, a.commit, a.source, a.model_desc)
    show = {k: v for k, v in agg.items() if k != "outcomes"}
    print(a.dest, {k: round(v, 4) if isinstance(v, float) else v for k, v in show.items()})
