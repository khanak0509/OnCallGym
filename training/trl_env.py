"""TRL GRPO adapter (environment_factory). Thin: every tool method is generated from
oncallgym.episode.Tools (same signature and docstring) and forwards to Episode.call_tool.

    GRPOTrainer(..., environment_factory=OnCallEnv)   # reward comes from get_reward()

Dataset rows carry family, variant and seed (see training/dataset.py).
"""

import functools

from oncallgym.episode import TOOL_NAMES, Episode, Tools

TOKENIZER = None  # set by the training script so the budget counts real tokens


def _count_tokens(text: str) -> int:
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


class OnCallEnv:
    def __init__(self):
        self.episode: Episode | None = None
        self.breakdown = None

    def reset(self, **row) -> str | None:
        self.close()
        self.breakdown = None
        self.episode = Episode(
            row["family"],
            int(row["variant"]),
            int(row["seed"]),
            count_tokens=_count_tokens if TOKENIZER is not None else None,
        )
        self.episode.start()  # the prompt itself comes from the dataset row (same text)
        return None

    def get_reward(self) -> float:
        if self.breakdown is None:
            self.breakdown = self.episode.finish()
            self.close()
        return self.breakdown.reward

    def close(self):
        if self.episode is not None:
            try:
                self.episode.close()
            finally:
                self.episode = None


def _make_delegate(name: str):
    target = getattr(Tools, name)

    @functools.wraps(target)
    def method(self, **kwargs):
        # arguments go to Episode unvalidated, exactly as in the eval path, so bad
        # arguments cost a turn and are logged the same way
        return self.episode.call_tool(name, kwargs)

    return method


for _name in TOOL_NAMES:
    setattr(OnCallEnv, _name, _make_delegate(_name))
