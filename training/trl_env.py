"""TRL GRPO adapter (environment_factory). Thin: every tool method is generated from
oncallgym.episode.Tools (same signature and docstring) and forwards to Episode.call_tool.

    GRPOTrainer(..., environment_factory=OnCallEnv)   # reward comes from get_reward()

Dataset rows carry family, variant and seed (training/dataset.py). Only scheduling differs
from eval, never semantics: the stack starts in a background thread during reset (the
prompt comes from the dataset row), and verification starts as soon as an episode ends so
it overlaps with the other rollouts. The episode, tools, limits and verifier are the same.
"""

import functools
import threading

from oncallgym.episode import TOOL_NAMES, Episode, Tools

TOKENIZER = None  # set by the training script (used for logging only; budget is shared)
_LOG: list[dict] = []
_LOG_LOCK = threading.Lock()


def drain_episode_log() -> list[dict]:
    with _LOG_LOCK:
        out = list(_LOG)
        _LOG.clear()
    return out


class OnCallEnv:
    def __init__(self):
        self.episode: Episode | None = None
        self._ready: threading.Thread | None = None
        self._start_error: Exception | None = None
        self._verify: threading.Thread | None = None
        self.breakdown = None

    def reset(self, **row) -> str | None:
        self.close()
        self.breakdown = None
        self._start_error = None
        self.episode = Episode(row["family"], int(row["variant"]), int(row["seed"]))

        def start():
            try:
                self.episode.start()
            except Exception as e:
                self._start_error = e

        self._ready = threading.Thread(target=start, daemon=True)
        self._ready.start()
        return None

    def _wait_ready(self):
        if self._ready is not None:
            self._ready.join()
            self._ready = None
        if self._start_error is not None:
            raise RuntimeError(f"episode failed to start: {self._start_error}")

    def _after_call(self):
        if self.episode.done and self._verify is None:
            self._verify = threading.Thread(target=self._finish, daemon=True)
            self._verify.start()

    def _finish(self):
        b = self.episode.finish()
        self.breakdown = b
        with _LOG_LOCK:
            _LOG.append(
                {
                    "variant": self.episode.fault.variant_id,
                    "reward": b.reward,
                    "R": b.R,
                    "P": b.P,
                    "C": b.C,
                    "G": b.G,
                    "termination": b.termination,
                    "turns": self.episode.turns,
                    "tokens_used": self.episode.tokens_used,
                }
            )

    def get_reward(self) -> float:
        if self.episode is None:
            return 0.0
        try:
            self._wait_ready()
        except RuntimeError:
            return 0.0  # infrastructure failure: logged by the caller, scored 0
        if self._verify is None:
            self._verify = threading.Thread(target=self._finish, daemon=True)
            self._verify.start()
        self._verify.join()
        reward = self.breakdown.reward
        self.close()
        return reward

    def close(self):
        if self._ready is not None:
            self._ready.join()
            self._ready = None
        if self._verify is not None:
            self._verify.join()
            self._verify = None
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
        self._wait_ready()
        out = self.episode.call_tool(name, kwargs)
        self._after_call()
        return out

    return method


for _name in TOOL_NAMES:
    setattr(OnCallEnv, _name, _make_delegate(_name))
