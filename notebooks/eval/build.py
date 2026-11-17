"""Bundle the repo into a Kaggle GPU kernel that runs the vf-run eval.

Usage: python build.py <kaggle-user> <out-dir> [model] [tasks-args] [rollouts]
"""

import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

here = Path(__file__).parent
root = here.parent.parent
user, dest = sys.argv[1], Path(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-1.7B"
tasks = sys.argv[4] if len(sys.argv) > 4 else "--env.taskset.variants 1"
rollouts = sys.argv[5] if len(sys.argv) > 5 else "1"
dest.mkdir(parents=True, exist_ok=True)
files = subprocess.run(
    [
        "git",
        "ls-files",
        "northwind",
        "oncallgym",
        "environments",
        "training",
        "pyproject.toml",
        "README.md",
    ],
    cwd=root,
    capture_output=True,
    text=True,
    check=True,
).stdout.split()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in files:
        t.add(root / f, arcname=f)
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
).stdout.strip()
src = (here / "eval_kernel.py").read_text()
for k, v in {
    "CODE_TGZ": base64.b64encode(buf.getvalue()).decode(),
    "COMMIT": commit,
    "MODEL": model,
    "TASKS": tasks,
    "ROLLOUTS": rollouts,
}.items():
    assert src.count(f"{k} = None") == 1, k
    src = src.replace(f"{k} = None", f"{k} = {v!r}")
(dest / "eval_kernel.py").write_text(src)
meta = {
    "id": f"{user}/oncallgym-eval",
    "title": "oncallgym-eval",
    "code_file": "eval_kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"built {dest} from {commit} ({len(files)} files), model={model}")
