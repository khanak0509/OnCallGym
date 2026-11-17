"""Kaggle GPU eval: serve a model with vLLM, run OnCallGym episodes through verifiers'
vf-run (the same eval path used for every reported number), save traces, trajectories and
a summary. Built by build.py (CODE_TGZ, COMMIT, MODEL, TASKS, ROLLOUTS filled in).
"""

import base64
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
MODEL = None
TASKS = None  # e.g. "--env.taskset.variants 1"
ROLLOUTS = None
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "work"
SRC = Path("/tmp/oncallgym-src")
VENVS = Path("/tmp/venvs")
PY = "3.12"
summary = {"commit": COMMIT, "model": MODEL, "stages": {}}


def save():
    (WORK / "eval_summary.json").write_text(json.dumps(summary, indent=1, default=str))


def sh(cmd, log, timeout=3600, env=None):
    t0 = time.time()
    with open(WORK / log, "w") as f:
        p = subprocess.run(
            cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, cwd=SRC, env=env
        )
    tail = (WORK / log).read_text(errors="replace")[-3000:]
    print(f"$ {cmd} -> {p.returncode} ({time.time() - t0:.0f}s)\n{tail[-800:]}", flush=True)
    return p.returncode, tail


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_TGZ)), mode="r:gz") as t:
        t.extractall(SRC, filter="data")
    t0 = time.time()
    sh(f"{sys.executable} -m pip install -q uv", "uv.log")
    vf, srv = VENVS / "vf", VENVS / "serve"
    rc, _ = sh(
        f"uv venv -q --python {PY} {vf} && uv pip install -q --python {vf}/bin/python "
        f"-e {SRC} -e {SRC}/environments/oncallgym_env",
        "vf_install.log",
        timeout=1800,
    )
    summary["stages"]["vf_install"] = rc
    rc, _ = sh(
        f"uv venv -q --python {PY} {srv} && uv pip install -q --python {srv}/bin/python "
        "vllm==0.33.0",
        "vllm_install.log",
        timeout=2400,
    )
    summary["stages"]["vllm_install"] = rc
    summary["install_s"] = round(time.time() - t0)
    save()
    rc, help_text = sh(
        f"{srv}/bin/vllm serve --help=all 2>&1 | grep -i -A2 chat-template-kwargs", "vllm_help.log"
    )
    kw = []
    if "default-chat-template-kwargs" in help_text:
        kw = ["--default-chat-template-kwargs", '{"enable_thinking": false}']
    summary["thinking_off_via"] = "server default" if kw else "not supported by server flag"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    log = open(WORK / "vllm_serve.log", "w")
    proc = subprocess.Popen(
        [
            f"{srv}/bin/vllm",
            "serve",
            MODEL,
            "--port",
            "8000",
            "--max-model-len",
            "8192",
            "--dtype",
            "float16",
            "--gpu-memory-utilization",
            "0.85",
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "hermes",
            "--served-model-name",
            "model",
            *kw,
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    for _ in range(240):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(5)
    summary["stages"]["vllm_up"] = proc.poll() is None
    save()
    if proc.poll() is not None:
        return
    # the agent's shell commands run as an unprivileged user
    sh(
        f"{vf}/bin/python -c 'from oncallgym.sandbox import ensure_agent_user, mode; "
        "ensure_agent_user(); print(mode())'",
        "sandbox.log",
    )
    traj = WORK / "trajectories"
    # Kaggle sets UV_SYSTEM_PYTHON, which breaks verifiers' `uv` harness script preparation
    run_env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    run_env.update(LOCAL_KEY="dummy", PATH=f"{vf}/bin:" + os.environ["PATH"])
    summary["removed_env"] = sorted(k for k in os.environ if k.startswith("UV_"))
    cmd = (
        f"{vf}/bin/vf-run oncallgym_env -m model --client.base-url http://127.0.0.1:8000/v1 "
        "--client.api-key-var LOCAL_KEY --env.agent.runtime.type subprocess "
        "--env.agent.harness.id null --env.agent.max-turns 20 "
        f"--env.taskset.task.trajectory-dir {traj} {TASKS} -r {ROLLOUTS} -c 3 "
        f"--output-dir {WORK}/vf_out"
    )
    t1 = time.time()
    rc, tail = sh(cmd, "vf_run.log", timeout=5400, env=run_env)
    summary["stages"]["vf_run"] = rc
    summary["vf_run_s"] = round(time.time() - t1)
    episodes = []
    for f in sorted(traj.glob("*.json")) if traj.exists() else []:
        events = json.loads(f.read_text())
        fin = next((e for e in events if e["kind"] == "finish"), None)
        episodes.append(
            {
                "file": f.name,
                "tool_calls": sum(e["kind"] == "tool" for e in events),
                "breakdown": fin
                and {
                    k: fin["breakdown"][k]
                    for k in ("R", "P", "C", "G", "reward", "solved", "termination")
                },
                "diagnosis": fin and fin["breakdown"]["details"].get("diagnosis"),
            }
        )
    summary["episodes"] = episodes
    proc.terminate()
    save()


try:
    main()
except Exception as e:
    summary["error"] = f"{type(e).__name__}: {e}"
finally:
    save()
    print(json.dumps(summary, indent=1, default=str)[:6000])
