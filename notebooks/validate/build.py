import base64
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

here = Path(__file__).parent
root = here.parent.parent
user, dest, args = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
dest.mkdir(parents=True, exist_ok=True)
files = subprocess.run(
    ["git", "ls-files", "northwind", "oncallgym", "pyproject.toml"],
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
src = (here / "validate_kernel.py").read_text()
for k, v in {
    "CODE_TGZ": base64.b64encode(buf.getvalue()).decode(),
    "COMMIT": commit,
    "ARGS": args,
}.items():
    assert src.count(f"{k} = None") == 1, k
    src = src.replace(f"{k} = None", f"{k} = {v!r}")
(dest / "validate_kernel.py").write_text(src)
meta = {
    "id": f"{user}/oncallgym-validate",
    "title": "oncallgym-validate",
    "code_file": "validate_kernel.py",
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
print(f"built {dest} from {commit} ({len(files)} files): {args}")
