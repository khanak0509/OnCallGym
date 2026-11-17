"""End to end through verifiers' vf-run with a scripted model (no GPU). Linux only; needs
the env package installed: uv pip install -e environments/oncallgym_env"""

import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import pytest

pytestmark = [
    pytest.mark.stack,
    pytest.mark.skipif(platform.system() != "Linux", reason="Linux stack only"),
]


def test_vf_run_scores_scripted_episode(tmp_path):
    pytest.importorskip("oncallgym_env")
    vf_run = shutil.which("vf-run") or os.path.join(os.path.dirname(sys.executable), "vf-run")
    port = 18765
    srv = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "fake_model.py"), str(port)]
    )
    try:
        time.sleep(1)
        traj = tmp_path / "traj"
        cmd = [
            vf_run,
            "oncallgym_env",
            "-m",
            "model",
            "--client.base-url",
            f"http://127.0.0.1:{port}/v1",
            "--client.api-key-var",
            "LOCAL_KEY",
            "--env.agent.runtime.type",
            "subprocess",
            "--env.agent.harness.id",
            "null",
            "--env.agent.max-turns",
            "10",
            "--env.taskset.families",
            '["jwt_key_rotation_mismatch"]',
            "--env.taskset.variants",
            "1",
            "--env.taskset.task.trajectory-dir",
            str(traj),
            "-n",
            "1",
            "-r",
            "1",
            "-c",
            "1",
            "-o",
            str(tmp_path / "out"),
        ]
        r = subprocess.run(
            cmd,
            env={**os.environ, "LOCAL_KEY": "dummy"},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert "1/1 episodes ok" in r.stdout + r.stderr, (r.stdout + r.stderr)[-3000:]
    finally:
        srv.kill()
    (f,) = glob.glob(str(traj / "*.json"))
    events = json.load(open(f))
    tools = [e["name"] for e in events if e["kind"] == "tool"]
    assert tools == ["list_services", "logs", "submit_diagnosis"]
    fin = next(e for e in events if e["kind"] == "finish")["breakdown"]
    # correct diagnosis, nothing fixed: only the root-cause component scores
    assert (fin["R"], fin["C"], fin["G"], fin["reward"]) == (0.0, 1.0, 1.0, 0.3)
    assert fin["termination"] == "submitted"
