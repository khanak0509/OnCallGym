"""SFT warm start (LoRA) on scripted oracle trajectories built by training/sft_data.py.

Rows are already tokenized exactly like GRPO rollouts (same template, schemas,
enable_thinking=False, tool-call format); the loss is on assistant tokens only
(labels -100 elsewhere). fp16 (T4 has no bf16).

    python training/sft.py --data data/sft_train.jsonl --out runs/sft --epochs 2
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
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--cpu", action="store_true", help="dry run on CPU (no fp16)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    rows = [json.loads(line) for line in open(a.data)]
    feats = []
    for r in rows:
        ids = r["input_ids"][: a.max_len]
        mask = r["assistant_mask"][: a.max_len]
        feats.append(
            {"input_ids": ids, "labels": [t if m else -100 for t, m in zip(ids, mask, strict=True)]}
        )
    ds = Dataset.from_list(feats).shuffle(seed=0)

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [0] * (n - len(b["input_ids"])) for b in batch]
        labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        attn = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    dtype = torch.float32 if a.cpu else torch.float16
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype)
    if not a.cpu:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    args = TrainingArguments(
        output_dir=a.out,
        num_train_epochs=a.epochs,
        max_steps=a.max_steps,
        learning_rate=a.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=a.grad_accum,
        fp16=not a.cpu,
        use_cpu=a.cpu,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * len(ds) * a.epochs / a.grad_accum)),
    )
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    model.save_pretrained(os.path.join(a.out, "adapter"))
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump(
            {
                "rows": len(rows),
                "seconds": round(time.time() - t0),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else None,
                "assistant_tokens": sum(sum(r["assistant_mask"]) for r in rows),
                "log_history": trainer.state.log_history,
                "args": vars(a),
            },
            f,
            indent=1,
        )


if __name__ == "__main__":
    main()
