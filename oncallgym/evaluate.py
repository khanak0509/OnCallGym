"""Evaluate a model on OnCallGym tasks with the shared loop (oncallgym.runner), the same
code path training uses. Official numbers come from this runner on Kaggle (vLLM, fp16).

    python -m oncallgym.evaluate --base-url http://127.0.0.1:8000 --backend vllm \\
        --model Qwen/Qwen3-1.7B --split test --variants 0-2 --samples 1 \\
        --out results/<run-name>

--backend llamacpp talks to llama.cpp's /completion (Mac dev runs only). Writes
<out>/trajectories/*.json (0600) and runs eval/summarize-style aggregation into
<out>/summary.json.
"""

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request

from .episode import fingerprint
from .runner import run_episodes
from .tasks import task_list


def _post(url, body, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def make_generate(base_url, backend, model, tokenizer, temperature, top_p, workers=8):
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def one(ids, max_new):
        if backend == "vllm":
            out = _post(
                f"{base_url}/v1/completions",
                {
                    "model": model,
                    "prompt": ids,
                    "max_tokens": max_new,
                    "temperature": temperature,
                    "top_p": top_p,
                    "skip_special_tokens": False,
                },
            )
            text = out["choices"][0]["text"]
        else:
            out = _post(
                f"{base_url}/completion",
                {
                    "prompt": ids,
                    "n_predict": max_new,
                    "temperature": temperature,
                    "top_p": top_p,
                    "cache_prompt": True,
                },
            )
            text = out["content"]
        new = tokenizer.encode(text.split("<|im_end|>")[0], add_special_tokens=False) + [eos]
        return new, [0.0] * len(new)

    def generate(batch, max_new):
        with cf.ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda ids: one(ids, max_new), batch))

    return generate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--backend", choices=["vllm", "llamacpp"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--split", default="all")
    ap.add_argument("--families", default="")
    ap.add_argument("--tiers", default="easy,medium,hard")
    ap.add_argument("--variants", default="0-0")
    ap.add_argument("--samples", type=int, default=1, help="episodes per task (pass@k)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from .faults import make_fault

    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    lo, hi = (int(x) for x in a.variants.split("-"))
    fams = [f for f in a.families.split(",") if f] or None
    tiers = set(a.tiers.split(","))
    specs = [
        {"family": f, "variant": v, "seed": v}
        for f, v in task_list(a.split, fams, variants=hi + 1)
        if v >= lo and make_fault(f, v).tier in tiers
        for _ in range(a.samples)
    ]
    gen = make_generate(a.base_url, a.backend, a.model, tok, a.temperature, a.top_p)
    traj_dir = os.path.join(a.out, "trajectories")
    os.makedirs(traj_dir, mode=0o700, exist_ok=True)
    t0 = time.time()
    results = []
    for i in range(0, len(specs), a.parallel):
        batch = run_episodes(specs[i : i + a.parallel], gen, tok)
        for j, r in enumerate(batch):
            path = os.path.join(traj_dir, f"{r['variant'].replace('/', '_')}-{i + j}.json")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                json.dump(r["trajectory"], f, default=str)
            results.append(r)
            b = r["breakdown"]
            print(
                f"{r['variant']:40s} reward={b['reward']:.2f} R={b['R']:.0f} P={b['P']:.0f} "
                f"C={b['C']:.0f} G={b['G']:.0f} {b['termination']}",
                flush=True,
            )
    meta = {
        "model": a.model,
        "backend": a.backend,
        "split": a.split,
        "variants": a.variants,
        "samples": a.samples,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "fingerprint": fingerprint(),
        "seconds": round(time.time() - t0),
        "episodes": len(results),
    }
    by_task = {}
    for r in results:
        by_task.setdefault(r["variant"], []).append(r["breakdown"])
    meta["pass_at_k"] = round(
        sum(any(b["solved"] for b in bs) for bs in by_task.values()) / max(len(by_task), 1), 4
    )
    meta["solve_rate"] = round(
        sum(b["solved"] for bs in by_task.values() for b in bs) / max(len(results), 1), 4
    )
    meta["mean_reward"] = round(
        sum(b["reward"] for bs in by_task.values() for b in bs) / max(len(results), 1), 4
    )
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
