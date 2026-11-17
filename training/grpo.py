"""GRPO (TRL) with LoRA on OnCallGym train families. Rollouts are real episodes through
training/trl_env.OnCallEnv (the same Episode, tools and verifier as eval).

    trl vllm-serve --model Qwen/Qwen3-1.7B ... (GPU 1)
    python training/grpo.py --out runs/grpo --adapter runs/sft/adapter --steps 60

Logs, per the owner's requirements for the first real run:
  - input length of every trainer forward pass (fails loudly if it is shorter than the
    rollouts' prompt + completion incl. tool tokens)
  - peak GPU memory per step, GradScaler scale and skipped (overflow) steps
  - termination reasons (context_exhausted etc.) and reward components per rollout
Checkpoints the adapter every --save-every steps so runs can resume across sessions.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="SFT adapter to start from")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--tiers", default="easy,medium,hard")
    ap.add_argument("--vllm-port", type=int, default=8000)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    import training.trl_env as trl_env
    from training.dataset import build_rows, to_hf_dataset

    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    trl_env.TOKENIZER = tok
    tiers = set(a.tiers.split(","))
    rows = [r for r in build_rows("train", variants=range(a.variants)) if r["tier"] in tiers]
    ds = to_hf_dataset(rows).shuffle(seed=0)

    log_path = os.path.join(a.out, "train_log.jsonl")
    forward_lengths: list[int] = []

    class Logger(TrainerCallback):
        def on_step_begin(self, args, state, control, **kw):
            self.t0 = time.time()
            torch.cuda.reset_peak_memory_stats()
            forward_lengths.clear()

        def on_log(self, args, state, control, logs=None, **kw):
            scaler = getattr(trainer.accelerator, "scaler", None)
            rec = {
                "step": state.global_step,
                "seconds": round(time.time() - getattr(self, "t0", time.time()), 1),
                "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "grad_scaler_scale": scaler.get_scale() if scaler is not None else None,
                "forward_input_lengths": list(forward_lengths),
                "episodes": trl_env.drain_episode_log(),
                **{k: v for k, v in (logs or {}).items()},
            }
            nonfinite = [k for k, v in rec.items() if isinstance(v, float) and not math.isfinite(v)]
            rec["nonfinite"] = nonfinite
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

        def on_step_end(self, args, state, control, **kw):
            if state.global_step % a.save_every == 0:
                trainer.save_model(os.path.join(a.out, f"checkpoint-{state.global_step}"))

    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    peft_config = None
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    cfg = GRPOConfig(
        output_dir=a.out,
        max_steps=a.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        num_generations=a.generations,
        max_completion_length=a.max_len - 1200,  # our Episode ends earlier (context_exhausted)
        max_tool_calling_iterations=a.max_turns,
        learning_rate=a.lr,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="server",
        vllm_server_port=a.vllm_port,
        chat_template_kwargs={"enable_thinking": False},
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        seed=0,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        environment_factory=trl_env.OnCallEnv,
        peft_config=peft_config,
        callbacks=[Logger()],
    )

    # record the real input length of every policy forward pass (owner requirement)
    orig = trainer._get_per_token_logps_and_entropies

    def logged(model, input_ids, *args, **kwargs):
        forward_lengths.append(int(input_ids.shape[-1]))
        return orig(model, input_ids, *args, **kwargs)

    trainer._get_per_token_logps_and_entropies = logged
    trainer.train(resume_from_checkpoint=a.resume)
    trainer.save_model(os.path.join(a.out, "adapter"))


if __name__ == "__main__":
    main()
