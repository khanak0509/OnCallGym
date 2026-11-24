"""Kaggle T4x2 training kernel: SFT warm start (GPU 0), then GRPO with real OnCallGym
episodes (trainer GPU 0, vLLM server GPU 1). Built by build.py with the code, the SFT data
and the run settings filled in. Everything it produces lands in /kaggle/working:
train_summary.json, sft/ (adapter + log), grpo/ (adapter, checkpoints, train_log.jsonl).
"""

import base64
import gzip
import io
import json
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

CODE_TGZ = None
COMMIT = None
SFT_DATA = None
SETTINGS = None
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "work"
SRC = Path("/tmp/oncallgym-src")
VENV = Path("/tmp/venvs/train")
PKGS = ["trl[vllm,peft]==1.18.0", "vllm==0.33.0", "datasets", "pyyaml"]
summary = {"commit": COMMIT, "settings": SETTINGS, "stages": {}}


def save():
    (WORK / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str))


def sh(cmd, log, timeout=7200, env=None):
    t0 = time.time()
    with open(WORK / log, "w") as f:
        p = subprocess.run(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=SRC, env=env
        )
    tail = (WORK / log).read_text(errors="replace")[-2500:]
    print(f"$ {cmd[:200]} -> {p.returncode} ({time.time() - t0:.0f}s)\n{tail[-600:]}", flush=True)
    return p.returncode, round(time.time() - t0)


def wait_up(port, proc, tries=240):
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            if proc.poll() is not None:
                return False
            time.sleep(5)
    return False


def find_adapter(path):
    """A LoRA adapter dir; for a dataset mount, search under it by the dataset's slug."""
    if Path(path, "adapter_config.json").exists():
        return path
    slug = Path(path).name
    found = sorted(
        f.parent for f in Path("/kaggle/input").rglob("adapter_config.json") if slug in str(f)
    )
    return str(found[0]) if found else path


def serve_and_eval(model, adapters, evals, py, env, prefix, tp=1, base=True):
    """Serve `model` with vLLM (optionally with LoRA adapters) and run the shared-loop
    evaluator for every eval config on every adapter (and the base model if `base`).
    Results land in WORK/{prefix}_{eval name}_{tag}; returns their meta.json contents."""
    log = open(WORK / f"vllm_{prefix}.log", "w")
    cmd = [
        f"{VENV}/bin/vllm",
        "serve",
        model,
        "--port",
        "8100",
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--tensor-parallel-size",
        str(tp),
    ]
    if adapters:
        cmd += ["--enable-lora", "--max-lora-rank", "16", "--lora-modules"]
        cmd += [f"{k}={v}" for k, v in adapters.items()]
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**env, "CUDA_VISIBLE_DEVICES": "0,1" if tp == 2 else "1"},
    )
    out = {"up": wait_up(8100, proc, tries=480)}
    if not out["up"]:
        return out
    names = list(adapters) + ([model] if base else [])
    for ev in evals:
        for name in names:
            tag = name if name in adapters else "base"
            key = f"{ev['name']}_{tag}" if ev.get("name") else tag
            try:
                rc, secs = sh(
                    f"{py} -m oncallgym.evaluate --base-url http://127.0.0.1:8100 "
                    f"--backend vllm --model {name} --tokenizer {model} "
                    f"--split {ev.get('split', 'train')} --tiers {ev.get('tiers', 'easy')} "
                    f"--variants {ev.get('variants', '0-3')} --samples {ev.get('samples', 8)} "
                    f"--parallel {ev.get('parallel', 8)} --out {WORK}/{prefix}_{key}",
                    f"{prefix}_{key}.log",
                    timeout=ev.get("timeout_s", 14400),
                    env={**env, "PYTHONPATH": str(SRC)},
                )
            except subprocess.TimeoutExpired:
                rc = "timeout"
            try:
                out[key] = json.loads((WORK / f"{prefix}_{key}" / "meta.json").read_text())
            except OSError:
                out[key] = {"rc": rc}
            summary["stages"][f"{prefix}_partial"] = out
            save()
    os.killpg(proc.pid, 15)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
    time.sleep(10)
    return out


def run_gate(s, gate, py, env, adapters, prefix="gate"):
    """As the gate, GRPO only starts if the SFT solve rate is within the band; after GRPO
    the same helper gives the before/after comparison. `gate` is one eval config, or has
    "evals": a list of them."""
    evals = gate.get("evals") or [gate]
    out = serve_and_eval(s["model"], adapters, evals, py, env, prefix, base=gate.get("base", True))
    rate = out.get("sft", {}).get("solve_rate", 0.0)
    lo, hi = gate.get("band", [0.10, 0.40])
    out["go"] = bool(lo <= rate <= hi) or bool(gate.get("force"))
    return out


def run_baselines(bl, py, env):
    """Open models of several sizes, no adapters, same evaluator and settings."""
    res = {}
    for m in bl["models"]:
        tag = m["model"].split("/")[-1].lower()
        res[tag] = serve_and_eval(m["model"], {}, bl["evals"], py, env, f"bl_{tag}", m.get("tp", 1))
        summary["stages"]["baselines"] = res
        save()
    return res


def latest_adapter(run_dir):
    final = Path(run_dir, "adapter")
    if Path(final, "adapter_config.json").exists():
        return str(final)
    ckpts = sorted(Path(run_dir).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_TGZ)), mode="r:gz") as t:
        t.extractall(SRC, filter="data")
    (SRC / "data").mkdir(exist_ok=True)
    (SRC / "data/sft_train.jsonl").write_bytes(gzip.decompress(base64.b64decode(SFT_DATA)))
    s = SETTINGS
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    sh(f"{sys.executable} -m pip install -q uv", "uv.log")
    pk = " ".join(f"'{p}'" for p in PKGS)
    rc, secs = sh(
        f"uv venv -q --python 3.12 {VENV} && uv pip install --python "
        f"{VENV}/bin/python {pk} && uv pip install --python {VENV}/bin/python "
        f"--no-deps -e {SRC}",
        "install.log",
        env=env,
    )
    summary["stages"]["install"] = {"rc": rc, "s": secs}
    save()
    py = f"{VENV}/bin/python"
    sh(
        f"{py} -c 'from oncallgym.sandbox import ensure_agent_user; "
        '[ensure_agent_user(f"agent{i}") for i in range(16)]\'',
        "sandbox.log",
    )
    gpu0 = {**env, "CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(SRC)}
    sft_adapter = s.get("sft_adapter") or f"{WORK}/sft/adapter"
    if s.get("sft_adapter"):
        sft_adapter = find_adapter(sft_adapter)
    summary["sft_adapter"] = sft_adapter
    if s.get("baselines"):
        run_baselines(s["baselines"], py, env)
        return
    ev = s.get("eval")
    if ev:
        adapters = {k: find_adapter(v) for k, v in ev.get("adapters", {}).items()}
        summary["eval_adapters"] = adapters
        summary["stages"]["eval"] = run_gate(s, ev, py, env, adapters, prefix="eval")
        save()
        return
    if s.get("sft", True) and not s.get("sft_adapter"):
        rc, secs = sh(
            f"{py} training/sft.py --data data/sft_train.jsonl --out {WORK}/sft "
            f"--epochs {s.get('sft_epochs', 2)}",
            "sft.log",
            env=gpu0,
        )
        summary["stages"]["sft"] = {"rc": rc, "s": secs}
        save()
        if rc:
            return
    gate = s.get("gate")
    if gate:
        passed = run_gate(s, gate, py, env, {"sft": sft_adapter})
        summary["stages"]["gate"] = passed
        save()
        if not passed.get("go"):
            return
    if not s.get("grpo_steps"):
        return
    log = open(WORK / "vllm.log", "w")
    vllm = subprocess.Popen(
        [
            f"{VENV}/bin/trl",
            "vllm-serve",
            "--model",
            s["model"],
            "--dtype",
            "float16",
            "--max_model_len",
            "8192",
            "--gpu_memory_utilization",
            "0.75",
            "--port",
            "8000",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**env, "CUDA_VISIBLE_DEVICES": "1"},
    )
    for _ in range(240):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health/", timeout=2)
            break
        except Exception:
            if vllm.poll() is not None:
                break
            time.sleep(5)
    summary["stages"]["vllm_up"] = vllm.poll() is None
    save()
    if vllm.poll() is not None:
        return
    adapter = f"--adapter {sft_adapter}" if s.get("sft", True) else ""
    runs = s.get("grpo_runs") or [{"name": "grpo", "args": ""}]
    for run in runs:
        t0 = time.time()
        try:
            rc, secs = sh(
                f"{py} training/grpo.py --out {WORK}/{run['name']} --model {s['model']} "
                f"{adapter} --steps {s['grpo_steps']} --variants {s.get('variants', 8)} "
                f"--tiers {s.get('tiers', 'easy,medium,hard')} {run['args']}",
                f"{run['name']}.log",
                timeout=s.get("grpo_timeout_s", 30000),
                env=gpu0,
            )
        except subprocess.TimeoutExpired:
            rc, secs = "timeout", round(time.time() - t0)
        summary["stages"][run["name"]] = {"rc": rc, "s": secs, "args": run["args"]}
        save()
    os.killpg(vllm.pid, 15)
    try:
        vllm.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(vllm.pid, 9)
    time.sleep(10)
    save()
    post = s.get("post_eval")
    grpo_adapter = latest_adapter(f"{WORK}/{runs[-1]['name']}")
    summary["grpo_adapter"] = grpo_adapter
    if post and grpo_adapter:
        adapters = {"grpo": grpo_adapter}
        if s.get("sft", True) and post.get("sft", True):
            adapters = {"sft": sft_adapter, **adapters}
        summary["stages"]["post_eval"] = run_gate(s, post, py, env, adapters, prefix="post")


try:
    main()
except Exception as e:
    summary["error"] = f"{type(e).__name__}: {e}"
finally:
    save()
    print(json.dumps(summary, indent=1, default=str)[:4000])
