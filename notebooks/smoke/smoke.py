"""Kaggle CPU smoke test for Northwind: unpack the bundled code, run the integration tests,
one probe and six stacks in parallel, and write smoke_summary.json.

Built by build.py, which replaces CODE_TGZ with the base64 tarball of the repo code.
"""

import base64
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time

CODE_TGZ = None
COMMIT = None
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
SRC = "/tmp/oncallgym-src"


def sh(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, shell=True, cwd=SRC, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout + p.stderr)[-12000:]
    print(f"$ {cmd}  (exit {p.returncode}, {time.time() - t0:.0f}s)\n{out}", flush=True)
    return p.returncode, out


def main():
    os.makedirs(SRC, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_TGZ)), mode="r:gz") as t:
        t.extractall(SRC, filter="data")
    summary = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
    }
    with open("/proc/meminfo") as f:
        summary["mem_total_kb"] = int(f.readline().split()[1])
    rc, out = sh(f"{sys.executable} -m pip install -q pyyaml pytest")
    rc, out = sh(f"{sys.executable} -m pytest -q tests 2>&1 | tail -15")
    summary["pytest"] = out.strip().splitlines()[-1] if out.strip() else ""
    rc, out = sh(f"{sys.executable} -m oncallgym.stack probe")
    try:
        probe = json.loads(out[out.index("{") :])
        summary["probe"] = {
            k: probe[k]
            for k in (
                "passed",
                "cold_start_s",
                "total_rss_mb",
                "recovery_window",
                "invariants",
                "restarts",
                "leftover_processes",
                "home_removed",
            )
        }
    except ValueError:
        summary["probe"] = {"error": out[-2000:]}
    load_before = os.getloadavg()
    rc, out = sh(f"{sys.executable} -m oncallgym.stack parallel 6 --warmup 20")
    try:
        summary["parallel6"] = json.loads(out[out.index("{") :])
    except ValueError:
        summary["parallel6"] = {"error": out[-2000:]}
    summary["loadavg_before_parallel"] = load_before
    summary["loadavg_after_parallel"] = os.getloadavg()
    rc, out = sh(f"{sys.executable} -m oncallgym.stack clean")
    summary["clean"] = out[-500:]
    with open(os.path.join(WORK, "smoke_summary.json"), "w") as f:
        json.dump({"commit": COMMIT, **summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


main()
