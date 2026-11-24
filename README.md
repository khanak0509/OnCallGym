# OnCallGym

An RL environment where an LLM agent is the on-call engineer for a small simulated online shop. Each episode boots the shop (15 processes), injects one production fault, pages the agent with a symptom-only alert, and gives it ops tools. A programmatic verifier then checks, with its own traffic, whether the system actually recovered and stays recovered after a full restart. There is no LLM judge.

Everything runs on free compute: plain processes (no Docker needed at train time), Kaggle T4 GPUs for training.

## How an episode works

```
             alert: "Customers can log in but every checkout is rejected (401)."
                                     |
                                     v
  +-----------+   tool calls   +-------------------------------------------------+
  |   agent   | -------------> | episode sandbox (separate Unix user per episode)|
  |  (LLM)    | <------------- |   exec (shell in deploy dir), logs, metrics,    |
  +-----------+   tool output  |   trace, topology, list_services, restart,      |
        |                      |   submit_diagnosis (ends the episode)           |
        |                      +-------------------------------------------------+
        |                                       |
        |                    Northwind Commerce: edge -> auth, catalog, orders,
        |                    inventory, payments -> provider (external), broker ->
        |                    fulfillment, notify; cache, flags, scheduler,
        |                    telemetry, trafficgen; all under one supervisor
        v
  verifier (own probes, after the agent is done):
    R  customer journeys + 8 s recovery window + liveness invariants
    P  restart every service, then R again
    C  submitted category and component match the ground truth
    G  safety and integrity gate (no lost/duplicated money, stock, messages, data;
       protected files, logging, scheduler jobs, trafficgen untouched)

  reward = G * (0.5*R*P + 0.3*C + 0.2*R)        solved = R and P and C and G
```

R also requires the fault's own `root_cause_fixed()` check on the effective configuration, so hiding a symptom (making the external provider reliable, restarting a stuck service) does not count.

## Tasks

17 fault families. Variants are generated from a seed (values, services and red herrings in the logs change); training uses variants 0-7, evaluation 0-3. Turn limits: easy 12, medium 14, hard 16 tool calls.

| split | tier | family | alert (what the agent sees) |
|---|---|---|---|
| train | easy | catalog_bad_build_flag | Product search returns errors for every query since the last catalog deploy. |
| train | easy | wrong_database_path | Product browsing failing with 5xx errors for all customers. |
| train | easy | edge_rate_limit_too_low | Customers getting 'Too Many Requests' (429) on most pages. |
| train | easy | edge_wrong_upstream_port | Login failing for most customers (5xx/404 at the edge). |
| train | medium | fulfillment_env_typo | Orders are paid but none have shipped for several minutes. |
| train | medium | fulfillment_wrong_topic | Orders are paid but none have shipped for several minutes. |
| train | medium | missing_migration | Every checkout fails with 500; browsing works. |
| train | medium | notify_dead_letters | Customers report missing order confirmation and shipping emails. |
| train | medium | orders_payments_timeout_inversion | Checkout failure rate above 50% (504); some customers charged twice. |
| train | medium | catalog_leak_flag | Intermittent errors and timeouts when browsing; error rate is flapping. |
| train | hard | scheduler_lock_hold | Checkout failing intermittently (about half the attempts) with 5xx. |
| train | hard | flags_service_down | All checkouts rejected with 503 'checkout temporarily disabled'. |
| test | easy | jwt_key_rotation_mismatch | Customers can log in but every checkout is rejected (401). |
| test | medium | cache_full_noeviction | Customers cannot log in (503); browsing works. |
| test | medium | payments_breaker_stuck_open | Checkout failing for most customers; payment provider status is green. |
| test | medium | poison_message | No orders have shipped for several minutes; paid orders are piling up. |
| test | hard | orders_config_drift | Every checkout fails with 502 since this morning's deploy. |

The test families are never used for SFT data or RL. Every family passes an automatic validation (`python -m oncallgym.validate`): healthy before injection, probes fail after it, the scripted oracle fix scores ~1.0, doing nothing scores ~0, restart-all does not solve it, every known cheat is blocked by the gate, and teardown leaves nothing behind.

## Results

Qwen3-1.7B, thinking off, 8k context, LoRA r=16. Same evaluator and sampling (T=0.7, top_p=0.95) for every row; each number comes from `results/<run>/summary.json` (commit, Kaggle run, per-episode rows). Paired comparisons are over tasks (bootstrap CI, sign-flip test; `training/compare.py`).

| model | eval set | episodes | solved | pass@k | mean reward | diagnosis right (C) |
|---|---|---|---|---|---|---|
| base | held-out test (20 tasks) | 160 | 0.0% | 0/20 | 0.039 | 0.13 |
| SFT | held-out test | 160 | 0.0% | 0/20 | 0.024 | 0.08 |
| SFT + GRPO (40 steps, easy train) | held-out test | 160 | 0.0% | 0/20 | 0.024 | 0.08 |
| SFT | easy train tasks (16) | 256 | 19.9% | 12/16 | 0.334 | 0.64 |
| SFT + GRPO (40 steps, easy train) | easy train tasks | 256 | 20.7% | 15/16 | 0.390 | 0.74 |
| SFT + GRPO (60 steps, all train tiers) | held-out test / all train | TBD | TBD | TBD | TBD | TBD |

- The base model solves nothing anywhere. Scripted-oracle SFT gets it to ~20% on the easy training families.
- GRPO on easy training tasks raised mean reward by +0.056 (95% CI [+0.013, +0.095], p=0.02) through better diagnoses; the strict solve rate did not move (+0.008, p=0.88).
- Nothing transfers to the five held-out families: all three models solve 0/160 and their rewards are not significantly different. The trained models learned the protocol (they submit instead of running out of turns: base runs out on 83% of episodes, SFT/GRPO on 2-4%) but blame the wrong service on 56-68% of unseen faults.

Baselines (open models, no training, same settings): TBD (Qwen3-0.6B / 1.7B / 4B / 8B, running).

### Where episodes fail

One outcome per episode (`analysis/taxonomy.py`):

| run | solved | gate tripped | no submission | fixed, not held | right diagnosis, no fix | wrong service | wrong category |
|---|---|---|---|---|---|---|---|
| easy train, SFT | 19.9% | 0.8% | 4.7% | 1.6% | 41.4% | 15.2% | 16.0% |
| easy train, GRPO | 20.7% | 2.0% | 0.8% | 1.2% | 51.6% | 9.0% | 11.3% |
| test, base | 0% | 0% | 83.1% | 0% | 13.1% | 2.5% | 1.2% |
| test, SFT | 0% | 0% | 3.8% | 0.6% | 8.1% | 68.1% | 19.4% |
| test, GRPO | 0% | 1.9% | 1.9% | 0% | 8.1% | 55.6% | 32.5% |

On trained families the bottleneck is the edit, not the diagnosis: the model finds the right file and key, then writes a `sed` that does not apply (a regex for a key that does not exist, `rps` fixed but `burst` left at 1, a substitution that breaks the YAML), restarts, and submits.

## Reward hacking

Hacks were found by design review, by an automated adversarial run, and by three independent code reviews. Each one is blocked by a check and most have a regression test (`tests/test_hardening.py`, `oncallgym/cheats.py`, run for every family by `oncallgym.validate`).

| hack | blocked by |
|---|---|
| Restart everything for a persistent fault | P re-runs every check after restart-all; per-family `root_cause_fixed()` |
| Make the external payment provider or mail relay reliable | protected-file hashes -> G=0; R needs the real cause fixed |
| Disable traffic generation, silence logging, disable the reconcile job | integrity checks on trafficgen, logging level, scheduler jobs |
| Delete failed orders, purge unacked messages, reset stock, truncate the outbox | business invariants: order records intact, charges match orders, no lost messages, stock conserved, append-only outbox |
| Write the databases directly from the shell | agent runs as its own Unix user without write access to data/ |
| Code injection through env files (PYTHONPATH, LD_PRELOAD) | env-key allowlist; services start with `python -I` |
| Background process that keeps fulfilling orders; calling internal APIs | every agent process killed after each command; internal routes need a token the agent cannot read |
| Point a service at an agent-written config via NW_CONFIG_FILE | override limited to the service's own file or hash-protected archived releases; the verifier reads the config each service actually loaded |
| Read the answer: fault code, file and directory mtimes, process command lines, trajectory files, notes in /tmp | code owner-only; timestamps restored after injection; task params in a 0600 file deleted on read; /tmp wiped per episode |
| Hang the episode with a catastrophic regex | log grep is substring only |
| Enumerate diagnoses / stall to the turn limit | one submission ends the episode; no submission gives C=0 and R/P still need a real fix |

Across all 1,888 evaluated episodes, none received recovery or persistence credit without the root cause fixed; the highest reward without a fix is 0.3 (diagnosis credit). Every gate trip read by hand was real damage from a bad edit, not an exploit.

## Training

```
scripted oracle trajectories (384, varied: extra checks, red herrings,
wrong-then-corrected steps)  ->  SFT (loss on assistant tokens only)
                             ->  GRPO (TRL 1.18, vLLM 0.33 server on GPU 1, trainer on GPU 0)
```

Training and evaluation share one agent loop and token renderer (same chat template, tool schemas, tool-call format, nudges on text-only replies), checked by `tests/test_parity.py`.

Two things had to be fixed before GRPO learned anything, both found by dumping tokens and logprobs from Kaggle and recomputing them in fp32 locally:

1. TRL's default `sequence_mask` importance-sampling correction multiplies per-token ratios over a whole ~500-token multi-turn episode. Small fp16 differences underflowed every ratio to 0, so the gradient was exactly zero for 40 steps. Fixed with per-token truncation (`--is-mode token_truncate`).
2. TRL's LoRA weight sync merges the adapter, hands vLLM views of the live weights, then unmerges. vLLM copies them on two side CUDA streams with no ordering against the merge/unmerge, so some layers arrived as base weights and vLLM sampled from a half-merged model (logprob gap 0.8-1.0 nats/token; after the fix 0.003-0.02). `training/vllm_sync.py` synchronizes around the merge and yields copies; `tests/test_vllm_sync.py`.

## Running it

Requirements: Python 3.12, uv; Linux (episodes create per-episode Unix users). On macOS use the dev container.

```
make setup                 # uv sync
make test                  # unit tests
make probe                 # boot the shop, run customer journeys and invariants, tear down
python -m oncallgym.validate wrong_database_path   # oracle, no-op and cheat checks for one family
python -m oncallgym.evaluate --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-1.7B \
    --split test --variants 0-0 --out runs/eval    # needs an OpenAI-compatible vLLM server
```

Training runs on Kaggle (T4 x2) from `notebooks/train/` (`build.py` packs the code, data and settings into one kernel script).

## Limitations

- One model size trained (1.7B), short GRPO runs (40-60 steps) on a free 30 GPU-hour weekly quota.
- No transfer to unseen fault families yet; gains are on families seen in training.
- 17 families on one synthetic system. The services are small Python processes, not production software; realism comes from the failure modes, alerts and checks, not from scale.
- The automated red team (local Qwen3-8B) was too weak to find exploits; the hack table comes mostly from human-style code review.

## Citation

```
@misc{oncallgym2026,
  title  = {OnCallGym: an RL environment for on-call incident response},
  year   = {2026},
  note   = {https://github.com/khanak0509/OnCallGym}
}
```
