"""Bundle the repo code into one Kaggle CPU script kernel.

Usage: python build.py <kaggle-username> <output-dir>
Then:  kaggle kernels push -p <output-dir>
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
dest.mkdir(parents=True, exist_ok=True)

files = subprocess.run(
    ["git", "ls-files", "northwind", "oncallgym", "tests", "pyproject.toml"],
    cwd=root,
    capture_output=True,
    text=True,
    check=True,
).stdout.split()
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in files:
        t.add(root / f, arcname=f)
code = base64.b64encode(buf.getvalue()).decode()
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True
).stdout.strip()

src = (here / "smoke.py").read_text()
assert src.count("CODE_TGZ = None") == 1 and src.count("COMMIT = None") == 1
src = src.replace("CODE_TGZ = None", f"CODE_TGZ = {code!r}")
src = src.replace("COMMIT = None", f"COMMIT = {commit!r}")
(dest / "smoke.py").write_text(src)
meta = {
    "id": f"{user}/oncallgym-smoke",
    "title": "oncallgym-smoke",
    "code_file": "smoke.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
(dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
print(f"built {dest} from commit {commit} ({len(files)} files, {len(code) // 1024} KB)")
