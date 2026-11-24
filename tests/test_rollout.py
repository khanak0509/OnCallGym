import json

from training.rollout import make_rollout_func


class FakeTrainer:
    num_generations = 4
    vllm_generation = None
    processing_class = None


def test_one_episode_per_prompt_same_task_per_group():
    rows = [
        {"prompt": [{"role": "user", "content": "A"}], "family": "fa", "variant": v, "seed": v}
        for v in range(3)
    ] + [{"prompt": [{"role": "user", "content": "B"}], "family": "fb", "variant": 0, "seed": 0}]
    seen = []

    def fake_run(specs, generate, tok, **kw):
        seen.extend(specs)
        return [
            {
                "prompt_ids": [1],
                "completion_ids": [2],
                "logprobs": [0.0],
                "env_mask": [1],
                "breakdown": {
                    "reward": 0.0,
                    "R": 0,
                    "P": 0,
                    "C": 0,
                    "G": 1,
                    "termination": "x",
                    "details": {},
                },
                "trajectory": [],
                "variant": f"{s['family']}/{s['variant']}",
            }
            for s in specs
        ]

    f = make_rollout_func(rows, run=fake_run)
    a, b = rows[0]["prompt"], rows[3]["prompt"]
    for _ in range(5):
        seen.clear()
        out = f([a] * 4 + [b] * 4, FakeTrainer())
        assert len(out["completion_ids"]) == 8 and len(seen) == 8
        assert len({json.dumps(s) for s in seen[:4]}) == 1
        assert all(s["family"] == "fb" for s in seen[4:])
