"""Spec 7.4 validation for fault variants.

For each variant, on fresh stacks:
  1 healthy before injection          probes pass
  2 probes fail after injection       the parts named in fault.breaks fail
  3 oracle fix -> solved, reward ~1   (with the correct diagnosis)
  4 do nothing -> reward ~0           (no diagnosis)
  5 restart all -> not solved         (unless fault.restart_fixable)
  6 cheats -> integrity gate G = 0    (generic + family cheats)
  7 invariants hold after the oracle  (part of 3: G = 1)
  8 teardown leaves nothing behind

    python -m oncallgym.validate [family ...] [--variants 0,1] [--skip-cheats]
"""

import argparse
import json
import os
import sys
import time

from .cheats import GENERIC_CHEATS
from .faults import FAMILIES, make_fault
from .probes import Probe
from .stack import Stack, leftover_processes
from .verifier import Baseline, verify


def _diag(fault):
    gt = fault.ground_truth
    return {
        "root_cause_category": gt.category,
        "component": gt.component,
        "summary": "validation oracle",
    }


def _episode(fault, seed, action):
    """Fresh stack, inject, run `action(stack)` -> diagnosis, verify. Returns breakdown."""
    with Stack(seed=seed) as s:
        time.sleep(2)
        fault.inject(s)
        baseline = Baseline.capture(s)
        time.sleep(fault.settle_s)
        diagnosis = action(s)
        b = verify(s, fault.ground_truth, diagnosis, baseline)
    return b


def _healthy_then_broken(family, variant, seed):
    fault = make_fault(family, variant)
    with Stack(seed=seed) as s:
        time.sleep(2)
        healthy = Probe(s).run(journeys=1, window_s=4)
        c1 = {"ok": healthy["passed"]}
        if not healthy["passed"]:
            c1["detail"] = healthy
        fault.inject(s)
        time.sleep(fault.settle_s)
        broken = Probe(s).run(journeys=1, window_s=6)
    from .probes import LIVENESS_INVARIANTS

    parts = {
        "journeys": not all(j["ok"] for j in broken["journeys"]),
        "window": not broken["recovery_window"]["ok"],
        "liveness": not all(
            v["ok"] for k, v in broken["invariants"].items() if k in LIVENESS_INVARIANTS
        ),
    }
    c2 = {
        "ok": all(parts[p] for p in fault.breaks if p in parts),
        "failed_parts": parts,
        "failed_step": next(
            (st for j in broken["journeys"] for st in j["steps"] if not st["ok"]), None
        ),
    }
    return {"1_healthy_before": c1, "2_probes_fail": c2}


def _run_episode(family, variant, seed, action):
    """action(stack, fault) -> diagnosis"""
    f = make_fault(family, variant)
    return _episode(f, seed, lambda s: action(s, f))


def _act_oracle(s, f):
    f.oracle(s)
    time.sleep(3)
    return _diag(f)


def _act_nothing(s, f):
    return None


def _act_restart_all(s, f):
    s.restart_all()
    time.sleep(3)
    return None


def validate_variant(
    family: str, variant: int, seed: int = 0, cheats: bool = True, workers: int = 3
) -> dict:
    import concurrent.futures as cf

    fault = make_fault(family, variant)
    out = {
        "variant": fault.variant_id,
        "category": fault.category,
        "component": fault.component,
        "tier": fault.tier,
        "checks": {},
    }
    checks = out["checks"]
    t0 = time.time()
    cheat_fns = {**GENERIC_CHEATS, **fault.cheats()} if cheats else {}
    applied = {name: {} for name in cheat_fns}

    def cheat_action(name, fn):
        def act(s, f):
            applied[name]["did"] = bool(fn(s, f))
            time.sleep(2)
            return _diag(f)

        return act

    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {
            "12": ex.submit(_healthy_then_broken, family, variant, seed),
            "oracle": ex.submit(_run_episode, family, variant, seed, _act_oracle),
            "nothing": ex.submit(_run_episode, family, variant, seed, _act_nothing),
            "restart": ex.submit(_run_episode, family, variant, seed, _act_restart_all),
        }
        for name, fn in cheat_fns.items():
            futs[f"cheat:{name}"] = ex.submit(
                _run_episode, family, variant, seed, cheat_action(name, fn)
            )
        res = {k: f.result() for k, f in futs.items()}

    checks.update(res["12"])
    b = res["oracle"]
    checks["3_oracle_solves"] = {
        "ok": b.solved and b.reward >= 0.99,
        "reward": b.reward,
        "R": b.R,
        "P": b.P,
        "C": b.C,
        "G": b.G,
        "details": b.details if not b.solved else None,
    }
    checks["7_invariants_after_oracle"] = {"ok": b.G == 1.0}
    b = res["nothing"]
    checks["4_do_nothing"] = {"ok": b.reward <= 0.05, "reward": b.reward, "R": b.R, "G": b.G}
    b = res["restart"]
    checks["5_restart_all"] = {
        "ok": (b.R == 1.0) == fault.restart_fixable,
        "reward": b.reward,
        "R": b.R,
        "restart_fixable": fault.restart_fixable,
    }
    if cheats:
        results = {}
        for name in cheat_fns:
            b = res[f"cheat:{name}"]
            integ = b.details["integrity"]
            results[name] = {
                "applied": applied[name].get("did"),
                "G": b.G,
                "reward": b.reward,
                "tripped": list(integ) if isinstance(integ, dict) else [],
            }
        checks["6_cheats_blocked"] = {
            "ok": all(r["G"] == 0.0 for r in results.values() if r["applied"]),
            "cheats": results,
        }
    left = leftover_processes()
    checks["8_teardown_clean"] = {"ok": not left, "leftover_pids": left}
    out["passed"] = all(c["ok"] for c in checks.values())
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--variants", default="0")
    ap.add_argument("--skip-cheats", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    variants = [int(v) for v in args.variants.split(",")]
    results = []
    for fam in args.families:
        for v in variants:
            r = validate_variant(fam, v, seed=v, cheats=not args.skip_cheats, workers=args.workers)
            results.append(r)
            short = {k: c["ok"] for k, c in r["checks"].items()}
            print(f"{r['variant']:45s} {'PASS' if r['passed'] else 'FAIL'} {short}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1, default=str)
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
