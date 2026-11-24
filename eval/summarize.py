import argparse
import datetime
import glob
import json
import os
import statistics
from collections import defaultdict


def load_episodes(traj_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(traj_dir, "*.json"))):
        events = json.load(open(f))
        start = next((e for e in events if e["kind"] == "start"), {})
        fin = next((e for e in events if e["kind"] == "finish"), None)
        tools = [e for e in events if e["kind"] == "tool"]
        eps.append(
            {
                "file": os.path.basename(f),
                "variant": start.get("variant"),
                "fingerprint": start.get("fingerprint"),
                "tool_calls": [t["name"] for t in tools],
                "finished": fin is not None,
                **(
                    {
                        k: fin["breakdown"][k]
                        for k in ("R", "P", "C", "G", "reward", "solved", "termination")
                    }
                    if fin
                    else {"reward": 0.0, "solved": False, "termination": "no_verify"}
                ),
                "diagnosis": fin and fin["breakdown"]["details"].get("diagnosis"),
                "ground_truth": fin and fin.get("ground_truth"),
            }
        )
    return eps


def aggregate(eps):
    def mean(xs):
        return round(statistics.mean(xs), 4) if xs else None

    by_family = defaultdict(list)
    for e in eps:
        by_family[(e["variant"] or "?").split("/")[0]].append(e)
    term = defaultdict(int)
    for e in eps:
        term[e["termination"]] += 1
    return {
        "episodes": len(eps),
        "mean_reward": mean([e["reward"] for e in eps]),
        "solved_rate": mean([float(e["solved"]) for e in eps]),
        **{f"{k}_rate": mean([e.get(k, 0.0) for e in eps]) for k in ("R", "P", "C", "G")},
        "mean_tool_calls": mean([len(e["tool_calls"]) for e in eps]),
        "terminations": dict(term),
        "by_family": {
            f: {
                "n": len(v),
                "mean_reward": mean([e["reward"] for e in v]),
                "solved_rate": mean([float(e["solved"]) for e in v]),
            }
            for f, v in sorted(by_family.items())
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_dir")
    ap.add_argument("run")
    ap.add_argument("--model", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    eps = load_episodes(a.traj_dir)
    fps = sorted({e["fingerprint"] for e in eps if e["fingerprint"]})
    summary = {
        "run": a.run,
        "model": a.model,
        "commit": a.commit,
        "date": datetime.date.today().isoformat(),
        "fingerprints": fps,
        "notes": a.notes,
        "aggregate": aggregate(eps),
        "episodes": eps,
    }
    os.makedirs(os.path.join(a.out, a.run), exist_ok=True)
    path = os.path.join(a.out, a.run, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=1)
    print(path)
    print(json.dumps(summary["aggregate"], indent=1))


if __name__ == "__main__":
    main()
