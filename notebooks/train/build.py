"""Usage: python build.py <kaggle-user> <out-dir> <sft-data.jsonl> '<settings json>'"""

import base64
import gzip
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

here = Path(__file__).parent
root = here.parent.parent
user, dest, data, settings = sys.argv[1], Path(sys.argv[2]), sys.argv[3], json.loads(sys.argv[4])
dest.mkdir(parents=True, exist_ok=True)
files = subprocess.run(
    ["git", "ls-files", "northwind", "oncallgym", "training", "pyproject.toml", "README.md"],
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
src = (here / "train_kernel.py").read_text()
for k, v in {
    "CODE_TGZ": base64.b64encode(buf.getvalue()).decode(),
    "COMMIT": commit,
    "SFT_DATA": base64.b64encode(gzip.compress(Path(data).read_bytes())).decode(),
    "SETTINGS": settings,
}.items():
    assert src.count(f"{k} = None") == 1, k
    src = src.replace(f"{k} = None", f"{k} = {v!r}")
(dest / "train_kernel.py").write_text(src)
meta = {
    "id": f"{user}/oncallgym-train",
    "title": "oncallgym-train",
    "code_file": "train_kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": settings.get("datasets", []),
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"built {dest} from {commit}, settings={settings}")
