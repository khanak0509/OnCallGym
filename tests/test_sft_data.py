import random

from oncallgym.faults import make_fault
from oncallgym.tasks import task_list
from training.sft_data import HERRING_SERVICES, WRONG_GUESSES, vary


def components(split):
    out = set()
    for f, v in task_list(split, variants=8):
        g = make_fault(f, v).ground_truth
        out.add(g.component)
        out.update(g.also_accepted)
    return out


def test_herrings_avoid_test_split_components():
    held_out = components("test")
    assert not set(HERRING_SERVICES) & held_out
    assert not set(WRONG_GUESSES) & held_out
    for cmd in WRONG_GUESSES.values():
        assert not any(c in cmd for c in held_out), cmd


def test_vary_never_dismisses_the_episode_component():
    steps = [("exec", {"command": "sed -i s/a/b/ etc/x.yaml"}, "fix")]
    for comp in HERRING_SERVICES:
        for seed in range(200):
            out = vary(steps, random.Random(seed), comp)
            for tool, args, _ in out:
                if tool == "logs":
                    assert args["service"] != comp
                if tool == "exec" and args["command"] != steps[0][1]["command"]:
                    assert args["command"] != WRONG_GUESSES.get(comp)
