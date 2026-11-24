import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import time

CODE_TGZ = None
COMMIT = None
ARGS = None
SRC = "/tmp/oncallgym-src"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()

os.makedirs(SRC, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_TGZ)), mode="r:gz") as t:
    t.extractall(SRC, filter="data")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=False)
t0 = time.time()
out = os.path.join(WORK, "validate.json")
p = subprocess.run(
    f"{sys.executable} -m oncallgym.validate {ARGS} --out {out}",
    shell=True,
    cwd=SRC,
    capture_output=True,
    text=True,
)
print(p.stdout[-20000:])
print(p.stderr[-8000:])
with open(os.path.join(WORK, "validate_meta.json"), "w") as f:
    json.dump(
        {
            "commit": COMMIT,
            "args": ARGS,
            "seconds": round(time.time() - t0),
            "exit": p.returncode,
            "cpus": os.cpu_count(),
        },
        f,
    )
