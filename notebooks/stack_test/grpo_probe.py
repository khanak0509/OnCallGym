"""Measure GRPO step time/memory and fp16 stability on one GPU.

Two modes, each run in its own process by stack_test.py:

  grpo    real TRL GRPO steps with multi-turn tool calls through a vLLM server
  memory  worst-case forward+backward of one full-length sequence with LoRA

The tool environment is synthetic (fake logs/metrics of a few services). Its only job
is to produce realistic multi-turn contexts; it is not the OnCallGym environment.
"""

import argparse
import json
import math
import os
import random
import time

import torch

SERVICES = ["edge", "auth", "catalog", "orders", "inventory", "payments", "broker", "notify"]
ROOT_CAUSE = "payments"

LOG_TEMPLATES = [
    "{ts} INFO  {svc} request_id={rid} method=GET path=/api/{svc}/items status=200 dur_ms={ms}",
    "{ts} INFO  {svc} request_id={rid} method=POST path=/api/{svc}/write status=201 dur_ms={ms}",
    "{ts} WARN  {svc} slow query table={svc}_main dur_ms={ms} rows=212",
    "{ts} DEBUG {svc} pool stats in_use=3 idle=5 waiting=0",
    "{ts} WARN  {svc} retrying upstream call attempt=2 after_ms=200",
]
FAULT_LINES = [
    "{ts} ERROR payments request_id={rid} provider call timed out after 9000ms",
    "{ts} ERROR payments circuit breaker OPEN failures=31 threshold=5",
]


def fake_logs(service: str, seed: int, lines: int = 14) -> str:
    rng = random.Random(f"{service}-{seed}")
    out = []
    for i in range(lines):
        ts = f"2026-11-16T10:{i // 6:02d}:{(i * 7) % 60:02d}.{rng.randint(0, 999):03d}Z"
        rid = f"{rng.getrandbits(48):012x}"
        template = rng.choice(LOG_TEMPLATES)
        if service == ROOT_CAUSE and rng.random() < 0.3:
            template = rng.choice(FAULT_LINES)
        out.append(template.format(ts=ts, svc=service, rid=rid, ms=rng.randint(3, 900)))
    return "\n".join(out)


class ProbeEnv:
    """Synthetic on-call environment exposing tools to TRL's environment_factory."""

    def __init__(self):
        self.seed = 0
        self.submitted = None

    def reset(self, **kwargs) -> str | None:
        self.seed = int(kwargs.get("seed", 0))
        self.submitted = None
        return None

    def list_services(self) -> str:
        """List all services with their status.

        Returns:
            One line per service with status and restart count.
        """
        return "\n".join(f"{s} status=running restarts=0 ready=true" for s in SERVICES)

    def logs(self, service: str) -> str:
        """Show the most recent log lines of a service.

        Args:
            service: Name of the service, for example "orders".

        Returns:
            The last log lines of the service.
        """
        if service not in SERVICES:
            return f"error: unknown service {service!r}"
        return fake_logs(service, self.seed)

    def metrics(self, service: str) -> str:
        """Show current metrics of a service.

        Args:
            service: Name of the service, for example "orders".

        Returns:
            Error rate, p95 latency and throughput.
        """
        if service not in SERVICES:
            return f"error: unknown service {service!r}"
        rng = random.Random(f"m-{service}-{self.seed}")
        err = 0.42 if service in (ROOT_CAUSE, "orders") else rng.random() * 0.01
        return json.dumps(
            {
                "service": service,
                "error_rate": round(err, 3),
                "p95_ms": rng.randint(20, 9000 if service == ROOT_CAUSE else 300),
                "rps": rng.randint(5, 80),
            }
        )

    def submit_diagnosis(self, component: str, category: str) -> str:
        """Submit the final diagnosis. Ends the investigation; call it exactly once.

        Args:
            component: The service that is the root cause.
            category: One of config, resilience_config, resource_exhaustion, dependency_down.

        Returns:
            Confirmation that the diagnosis was recorded.
        """
        if self.submitted is None:
            self.submitted = (component.strip().lower(), category.strip().lower())
        return "Diagnosis recorded. The episode is over; do not call more tools."

    def get_reward(self) -> float:
        if self.submitted is None:
            return 0.0
        return 1.0 if self.submitted[0] == ROOT_CAUSE else 0.1


PROMPT = (
    "You are the on-call engineer. Users report that checkout fails with 502 errors. "
    "Investigate with the tools: check the logs and metrics of several services before deciding. "
    "When you are confident, call submit_diagnosis once."
)


def model_kwargs(name: str) -> dict:
    kwargs = {}
    if "Instruct-2507" not in name:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    return kwargs


def run_grpo(args) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    records = []

    class StepTimer(TrainerCallback):
        def on_step_begin(self, a, state, control, **kw):
            self.t0 = time.time()
            torch.cuda.reset_peak_memory_stats()

        def on_step_end(self, a, state, control, **kw):
            records.append(
                {
                    "step": state.global_step,
                    "seconds": round(time.time() - self.t0, 1),
                    "peak_gb_gpu0": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
            )

        def on_log(self, a, state, control, logs=None, **kw):
            if logs and records:
                records[-1].setdefault("logs", {}).update(logs)

    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": PROMPT}], "seed": i} for i in range(64)]
    )
    config = GRPOConfig(
        output_dir=os.path.join(args.out, "grpo_tmp"),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.generations,
        max_completion_length=args.max_len - 1024,
        max_tool_calling_iterations=args.max_turns,
        learning_rate=1e-5,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        model_init_kwargs={"dtype": "float16"},
        use_vllm=True,
        vllm_mode="server",
        vllm_server_port=args.vllm_port,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        seed=0,
        **model_kwargs(args.model),
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules="all-linear"
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        environment_factory=ProbeEnv,
        peft_config=peft_config,
        callbacks=[StepTimer()],
    )
    t0 = time.time()
    trainer.train()
    total = time.time() - t0

    def finite(rec, key):
        value = rec.get("logs", {}).get(key)
        return value is None or math.isfinite(value)

    unstable = [r["step"] for r in records if not (finite(r, "loss") and finite(r, "grad_norm"))]
    return {
        "mode": "grpo",
        "total_seconds": round(total, 1),
        "steps": records,
        "nonfinite_steps": unstable,
    }


def run_memory(args) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map={"": 0}
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model, LoraConfig(r=16, lora_alpha=32, task_type="CAUSAL_LM", target_modules="all-linear")
    )
    results = []
    for seq_len in (4096, args.max_len):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, 50000, (1, seq_len), device="cuda:0")
        completion = seq_len - 1024
        t0 = time.time()
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=ids, logits_to_keep=completion + 1, use_cache=False)
                logits = out.logits[:, :-1]
                target = ids[:, -completion:]
                # chunked log-softmax so the fp32 copy of the logits never exists at once
                token_logps = []
                for start in range(0, completion, 1024):
                    chunk = torch.log_softmax(logits[:, start : start + 1024].float(), dim=-1)
                    tgt = target[:, start : start + 1024].unsqueeze(-1)
                    token_logps.append(chunk.gather(-1, tgt).squeeze(-1))
                loss = -torch.cat(token_logps, dim=1).mean()
            loss.backward()
            torch.cuda.synchronize()
            results.append(
                {
                    "seq_len": seq_len,
                    "ok": True,
                    "seconds": round(time.time() - t0, 2),
                    "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                    "loss_finite": bool(torch.isfinite(loss).item()),
                }
            )
        except torch.OutOfMemoryError as e:
            results.append({"seq_len": seq_len, "ok": False, "error": str(e)[:300]})
        model.zero_grad(set_to_none=True)
    return {"mode": "memory", "results": results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["grpo", "memory"])
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--vllm-port", type=int, default=8000)
    args = p.parse_args()

    result = {
        "model": args.model,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "args": vars(args),
    }
    try:
        result.update(run_grpo(args) if args.mode == "grpo" else run_memory(args))
        result["ok"] = True
    except Exception:  # recorded, not hidden: the orchestrator reports it
        import traceback

        result["ok"] = False
        result["error"] = traceback.format_exc()[-3000:]
    tag = args.model.split("/")[-1]
    with open(os.path.join(args.out, f"{args.mode}_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, default=str)[:2000])


if __name__ == "__main__":
    main()
