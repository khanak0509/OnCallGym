"""Kaggle stack test: what does the real T4 x2 image give us, and what does one GRPO step cost?

Runs top to bottom on a Kaggle GPU (T4 x2) session with internet on. Everything lands in
/kaggle/working/stack_test/: summary.json plus per-stage logs. Nothing here touches any
account credentials.

Stages
  1. environment report (GPU, driver, CUDA, Python, preinstalled torch/transformers, CPU, RAM, disk)
  2. verifiers==0.4.1 in its own venv, import check
  3. training stack (trl + vllm + peft) in its own venv, resolved versions recorded
  4. per model: worst-case memory probe on GPU0, then real GRPO steps with vLLM on GPU1
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path("/kaggle/working/stack_test")
LOGS = OUT / "logs"
VENVS = Path("/tmp/venvs")
MODELS = ["Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-1.7B"]
TRAIN_PKGS = ["trl[vllm,peft]==1.18.0", "vllm==0.33.0", "datasets"]
PY = "3.12"

# replaced by build.py with the contents of grpo_probe.py
PROBE_SRC = None

summary: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def save():
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))


def sh(cmd: str, log: str, env: dict | None = None, timeout: int = 3600) -> tuple[int, float, str]:
    """Run a shell command, tee to a log file, return (code, seconds, last lines)."""
    t0 = time.time()
    path = LOGS / log
    with open(path, "w") as f:
        try:
            code = subprocess.run(
                cmd,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, **(env or {})},
                timeout=timeout,
            ).returncode
        except subprocess.TimeoutExpired:
            code = -9
    tail = "\n".join(path.read_text(errors="replace").splitlines()[-25:])
    print(f"[{code}] {cmd[:120]} ({time.time() - t0:.0f}s)", flush=True)
    return code, round(time.time() - t0, 1), tail


def tail_of(path: Path, n: int) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def out(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def stage_env():
    info = {
        "nvidia_smi": out(
            "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap "
            "--format=csv,noheader"
        ),
        "cuda_runtime_banner": out("nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
        "nvcc": out("nvcc --version | tail -1"),
        "python": sys.version,
        "preinstalled": out(
            f"{sys.executable} -m pip list 2>/dev/null | grep -iE "
            "'^(torch|transformers|vllm|trl|peft|bitsandbytes|accelerate|unsloth) '",
        ),
        "cpu_count": os.cpu_count(),
        "mem": out("free -g | head -2"),
        "disk_working": out("df -h /kaggle/working | tail -1"),
        "disk_tmp": out("df -h /tmp | tail -1"),
        "docker": out("which docker || echo none"),
    }
    summary["stages"]["env"] = info


def stage_uv():
    code, secs, tail = sh(f"{sys.executable} -m pip install -q uv", "uv.log")
    summary["stages"]["uv"] = {"code": code, "seconds": secs, "version": out("uv --version")}


def stage_verifiers():
    venv = VENVS / "vf"
    steps = [
        sh(f"uv venv -q --python {PY} {venv}", "vf_venv.log"),
        sh(f"uv pip install -q --python {venv}/bin/python verifiers==0.4.1", "vf_install.log"),
        sh(
            f"{venv}/bin/python -c 'import verifiers, verifiers.v1; print(verifiers.__version__)'",
            "vf_import.log",
        ),
        sh(f"{venv}/bin/vf-run --help | head -5", "vf_run_help.log"),
    ]
    summary["stages"]["verifiers"] = {
        "ok": all(s[0] == 0 for s in steps),
        "seconds": sum(s[1] for s in steps),
        "failures": [s[2] for s in steps if s[0] != 0],
    }


def stage_train_stack():
    venv = VENVS / "train"
    pkgs = " ".join(f"'{p}'" for p in TRAIN_PKGS)
    c1 = sh(f"uv venv -q --python {PY} {venv}", "train_venv.log")
    c2 = sh(f"uv pip install --python {venv}/bin/python {pkgs}", "train_install.log", timeout=2400)
    versions = out(
        f"uv pip list --python {venv}/bin/python 2>/dev/null | grep -iE "
        "'^(torch|transformers|vllm|trl|peft|accelerate|datasets|triton|xformers|flashinfer)'"
    )
    cuda = out(
        f"{venv}/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, "
        "torch.cuda.is_available(), torch.cuda.device_count())'"
    )
    summary["stages"]["train_stack"] = {
        "ok": c1[0] == 0 and c2[0] == 0,
        "install_seconds": c2[1],
        "versions": versions,
        "torch_cuda": cuda,
        "failure": c2[2] if c2[0] else None,
    }
    return summary["stages"]["train_stack"]["ok"]


def wait_health(port: int, proc: subprocess.Popen, timeout: int = 1200) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def start_vllm(model: str, eager: bool) -> tuple[subprocess.Popen, Path]:
    tag = model.split("/")[-1]
    log = LOGS / f"vllm_{tag}{'_eager' if eager else ''}.log"
    cmd = [
        str(VENVS / "train/bin/trl"),
        "vllm-serve",
        "--model",
        model,
        "--dtype",
        "float16",
        "--max_model_len",
        "8192",
        "--gpu_memory_utilization",
        "0.85",
        "--port",
        "8000",
    ]
    if eager:
        cmd += ["--enforce_eager", "true"]
    f = open(log, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
    )
    return proc, log


def stop(proc: subprocess.Popen):
    if proc.poll() is None:
        os.killpg(proc.pid, 15)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)


def stage_model(model: str, probe: Path):
    tag = model.split("/")[-1]
    py = VENVS / "train/bin/python"
    res: dict = {}
    gpu0 = {"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    code, secs, tail = sh(
        f"{py} {probe} memory --model {model} --out {OUT}", f"memory_{tag}.log", env=gpu0
    )
    res["memory_probe"] = {"code": code, "seconds": secs, "tail": tail if code else None}

    for eager in (False, True):
        proc, log = start_vllm(model, eager)
        t0 = time.time()
        healthy = wait_health(8000, proc)
        res[f"vllm_{'eager' if eager else 'graphs'}"] = {
            "healthy": healthy,
            "startup_seconds": round(time.time() - t0, 1),
            "tail": None if healthy else tail_of(log, 30),
        }
        if not healthy:
            stop(proc)
            continue
        code, secs, tail = sh(
            f"{py} {probe} grpo --model {model} --out {OUT} --steps 4",
            f"grpo_{tag}.log",
            env=gpu0,
            timeout=5400,
        )
        res["grpo"] = {
            "code": code,
            "seconds": secs,
            "cuda_graphs": not eager,
            "tail": tail if code else None,
        }
        res["vllm_log_tail"] = "\n".join(log.read_text(errors="replace").splitlines()[-15:])
        stop(proc)
        break
    summary["stages"][f"model_{tag}"] = res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    VENVS.mkdir(parents=True, exist_ok=True)
    probe = OUT / "grpo_probe.py"
    probe.write_text(PROBE_SRC)

    for fn in (stage_env, stage_uv, stage_verifiers):
        fn()
        save()
    if stage_train_stack():
        save()
        for model in MODELS:
            stage_model(model, probe)
            save()
    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    shutil.rmtree(OUT / "grpo_tmp", ignore_errors=True)
    print(json.dumps(summary, indent=2, default=str)[:20000])


if __name__ == "__main__":
    main()
