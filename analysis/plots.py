"""Figures for the README and paper, generated only from results/*/summary.json.

python analysis/plots.py          # writes results/plots/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "plots"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated, fixed order

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    }
)


def load(name):
    return json.loads((RES / name / "summary.json").read_text())


def sync_fix():
    d = load("grpo_sync_fix")["steps"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for key, label, color in [
        ("racy_sync", "TRL LoRA sync (racy)", SERIES[1]),
        ("fixed_sync", "synchronized sync (training/vllm_sync.py)", SERIES[0]),
    ]:
        xs = [r["step"] for r in d[key]]
        ys = [r["logp_gap_mean"] for r in d[key]]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=3.5, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("mean |trainer - vLLM| logprob\n(nats per sampled token, log scale)")
    ax.set_title("vLLM samples from the trainer's weights only after the sync fix", loc="left")
    ax.legend(frameon=False, loc="center right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "sync_fix_logprob_gap.png", dpi=160)
    plt.close(fig)


# taxonomy outcomes folded into five groups for the figure (full split in the README table)
GROUPS = [
    ("solved", ["solved"]),
    ("fixed, not solved", ["fix_not_held", "fixed_wrong_diag"]),
    ("right diagnosis, no fix", ["diag_no_fix"]),
    ("wrong diagnosis", ["wrong_component", "wrong_category"]),
    ("no submission / gate", ["no_submission", "guardrail"]),
]
RUNS = [
    ("easy train · SFT", "train-easy_sft"),
    ("easy train · SFT+GRPO", "train-easy_grpo-easy40"),
    ("held-out · base", "test_base-1.7b"),
    ("held-out · SFT", "test_sft"),
    ("held-out · SFT+GRPO", "test_grpo-easy40"),
]


def outcomes():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [r[0] for r in RUNS][::-1]
    data = [load(r[1])["aggregate"]["outcomes"] for r in RUNS][::-1]
    left = [0.0] * len(data)
    for (gname, keys), color in zip(GROUPS, SERIES, strict=True):
        vals = [100 * sum(o[k] for k in keys) for o in data]
        ax.barh(
            labels,
            vals,
            left=left,
            color=color,
            edgecolor=SURFACE,
            linewidth=2,
            height=0.62,
            label=gname,
        )
        for i, v in enumerate(vals):
            if v >= 8:
                ax.text(
                    left[i] + v / 2,
                    i,
                    f"{v:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=INK,
                )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of episodes (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Where episodes end (Qwen3-1.7B)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.22), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "outcomes.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sync_fix()
    outcomes()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
