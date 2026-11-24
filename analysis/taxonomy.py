import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ORDER = [
    "solved",
    "guardrail",
    "no_submission",
    "fix_not_held",
    "fixed_wrong_diag",
    "diag_no_fix",
    "wrong_component",
    "wrong_category",
]


def outcome(b):
    d = b.get("details") or {}
    rc = d.get("root_cause") or {}
    fixed = bool(d.get("root_cause_fixed"))
    diag_ok = bool(rc.get("category_ok") and rc.get("component_ok"))
    if b.get("solved"):
        return "solved"
    if b.get("G", 1) < 1:
        return "guardrail"
    if b.get("termination") != "submitted":
        return "no_submission"
    if fixed and not (b.get("R") and b.get("P")):
        return "fix_not_held"
    if fixed:
        return "fixed_wrong_diag"
    if diag_ok:
        return "diag_no_fix"
    if not rc.get("component_ok"):
        return "wrong_component"
    return "wrong_category"


def load(d):
    rows = []
    for f in sorted(Path(d, "trajectories").glob("*.json")):
        ev = json.loads(f.read_text())
        fin = next((e for e in reversed(ev) if e.get("kind") == "finish"), None)
        if fin is None:
            continue
        b = fin["breakdown"]
        family = f.stem.rsplit("-", 1)[0].rsplit("_", 1)[0]
        tier = None
        try:
            from oncallgym.faults import make_fault

            tier = make_fault(family, 0).tier
        except Exception:
            pass
        rows.append({"family": family, "tier": tier, "outcome": outcome(b), "reward": b["reward"]})
    return rows


def main(argv):
    md = js = None
    runs = []
    it = iter(argv)
    for x in it:
        if x == "--md":
            md = next(it)
        elif x == "--json":
            js = next(it)
        else:
            name, path = x.split("=", 1)
            runs.append((name, path))
    table = {}
    by_family = {}
    for name, path in runs:
        rows = load(path)
        c = Counter(r["outcome"] for r in rows)
        n = len(rows)
        table[name] = {"n": n, **{k: c.get(k, 0) / n if n else 0 for k in ORDER}}
        fam = defaultdict(Counter)
        for r in rows:
            fam[r["family"]][r["outcome"]] += 1
        by_family[name] = {f: dict(v) for f, v in sorted(fam.items())}
    lines = ["| run | n | " + " | ".join(ORDER) + " |", "|---|---|" + "---|" * len(ORDER)]
    for name, t in table.items():
        lines.append(
            f"| {name} | {t['n']} | " + " | ".join(f"{100 * t[k]:.1f}%" for k in ORDER) + " |"
        )
    out = "\n".join(lines)
    print(out)
    if md:
        Path(md).write_text(out + "\n")
    if js:
        Path(js).write_text(json.dumps({"table": table, "by_family": by_family}, indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main(sys.argv[1:])
