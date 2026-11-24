import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"


def main():
    n = credit_without_fix = 0
    max_unfixed = 0.0
    runs = []
    for f in sorted(RES.glob("*/summary.json")):
        eps = json.loads(f.read_text()).get("episodes")
        if not isinstance(eps, list):
            continue
        if eps and "root_cause_fixed" not in eps[0]:
            print(f"skipped {f.parent.name}: older format without root_cause_fixed")
            continue
        runs.append(f.parent.name)
        for e in eps:
            n += 1
            if not e["root_cause_fixed"]:
                credit_without_fix += bool(e["R"] or e["P"])
                max_unfixed = max(max_unfixed, e["reward"])
    print(f"{len(runs)} eval runs, {n} episodes")
    print(f"R or P credit without the root cause fixed: {credit_without_fix}")
    print(f"highest reward without the root cause fixed: {max_unfixed:.2f}")


if __name__ == "__main__":
    main()
