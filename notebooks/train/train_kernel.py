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


def run_gate(s, gate, py, env):
    """pass@k of base and SFT models on easy train tasks with the shared-loop evaluator.
    GRPO only starts if the SFT model's per-sample solve rate is within the band."""
    log = open(WORK / "vllm_gate.log", "w")
    cmd = [
        f"{VENV}/bin/vllm",
        "serve",
        s["model"],
        "--port",
        "8100",
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-lora",
        "--max-lora-rank",
        "16",
        "--lora-modules",
        f"sft={WORK}/sft/adapter",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**env, "CUDA_VISIBLE_DEVICES": "1"},
    )
    out = {"up": wait_up(8100, proc)}
    if not out["up"]:
        return out
    for name in ("sft", s["model"]) if gate.get("base", True) else ("sft",):
        tag = "sft" if name == "sft" else "base"
        rc, secs = sh(
            f"{py} -m oncallgym.evaluate --base-url http://127.0.0.1:8100 "
            f"--backend vllm --model {name} --tokenizer {s['model']} "
            f"--split {gate.get('split', 'train')} --tiers {gate.get('tiers', 'easy')} "
            f"--variants {gate.get('variants', '0-3')} --samples {gate.get('samples', 8)} "
            f"--parallel 8 --out {WORK}/gate_{tag}",
            f"gate_{tag}.log",
            env={**env, "PYTHONPATH": str(SRC)},
        )
        try:
            out[tag] = json.loads((WORK / f"gate_{tag}" / "meta.json").read_text())
        except OSError:
            out[tag] = {"rc": rc}
    proc.terminate()
    rate = out.get("sft", {}).get("solve_rate", 0.0)
    lo, hi = gate.get("band", [0.10, 0.40])
    out["go"] = bool(lo <= rate <= hi) or bool(gate.get("force"))
    return out


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
    if s.get("sft", True):
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
        passed = run_gate(s, gate, py, env)
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
    adapter = f"--adapter {WORK}/sft/adapter" if s.get("sft", True) else ""
    rc, secs = sh(
        f"{py} training/grpo.py --out {WORK}/grpo --model {s['model']} {adapter} "
        f"--steps {s['grpo_steps']} --variants {s.get('variants', 8)} "
        f"--tiers {s.get('tiers', 'easy,medium,hard')}",
        "grpo.log",
        timeout=s.get("grpo_timeout_s", 30000),
        env=gpu0,
    )
    summary["stages"]["grpo"] = {"rc": rc, "s": secs}
    vllm.terminate()


try:
    main()
except Exception as e:
    summary["error"] = f"{type(e).__name__}: {e}"
finally:
    save()
    print(json.dumps(summary, indent=1, default=str)[:4000])
