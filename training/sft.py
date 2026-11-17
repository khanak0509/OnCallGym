"""SFT warm start (LoRA) on scripted oracle trajectories (data/sft_train.jsonl, built by
training/sft_data.py from our own scripted solutions; no model-generated data).

    python training/sft.py --data data/sft_train.jsonl --out runs/sft --epochs 2

Teaches the tool-call format and the investigate -> fix -> submit loop before RL. Loss is
on assistant turns only. fp16 (T4 has no bf16). Writes the adapter and train_log.json.
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--cpu", action="store_true", help="dry run on CPU (no fp16)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(line) for line in open(a.data)]
    # per-row template arguments (TRL reads the chat_template_kwargs column)
    ds = Dataset.from_list(
        [
            {
                "messages": r["messages"],
                "tools": r["tools"],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            for r in rows
        ]
    )
    cfg = SFTConfig(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum,
        max_length=a.max_len,
        assistant_only_loss=True,
        max_steps=a.max_steps,
        fp16=not a.cpu,
        bf16=False,
        use_cpu=a.cpu,
        gradient_checkpointing=not a.cpu,
        model_init_kwargs={"dtype": "float32" if a.cpu else "float16"},
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
    )
    peft = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear"
    )
    t0 = time.time()
    trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=ds, peft_config=peft)
    trainer.train()
    trainer.save_model(os.path.join(a.out, "adapter"))
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump(
            {
                "rows": len(rows),
                "seconds": round(time.time() - t0),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else None,
                "log_history": trainer.state.log_history,
                "args": vars(a),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
