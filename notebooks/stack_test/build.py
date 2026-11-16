"""Bundle stack_test.py + grpo_probe.py into one Kaggle script kernel folder.

Usage: python build.py <kaggle-username> <output-dir> [plan]
"""

import json
import sys
from pathlib import Path

here = Path(__file__).parent
user, dest = sys.argv[1], Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)

plan = sys.argv[3] if len(sys.argv) > 3 else "full"
src = (here / "stack_test.py").read_text()
assert src.count('PLAN = "full"') == 1
src = src.replace('PLAN = "full"', f"PLAN = {plan!r}")
probe = (here / "grpo_probe.py").read_text()
marker = "PROBE_SRC = None"
assert src.count(marker) == 1
(dest / "stack_test.py").write_text(src.replace(marker, f"PROBE_SRC = {probe!r}"))

meta = {
    "id": f"{user}/oncallgym-stack-test",
    "title": "oncallgym-stack-test",
    "code_file": "stack_test.py",
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
print(f"built {dest}")
