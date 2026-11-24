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


def fake_logs(service: str, seed: int, lines: int = 16) -> str:
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


TOKENIZER = None
ROLLOUTS: list[dict] = []
TOOL_TOKENS = 0


def pad_to(text: str, target: int, make_line) -> str:
    if target <= 0 or TOKENIZER is None:
        return text
    lines = [text]
    i = 0
    while len(TOKENIZER("\n".join(lines))["input_ids"]) < target:
        lines.append(make_line(i))
        i += 1
    return "\n".join(lines)


class ProbeEnv:
    def __init__(self):
        self.seed = 0
        self.submitted = None
        self.tool_tokens = 0
        self.calls = 0

    def reset(self, **kwargs) -> str | None:
        self.seed = int(kwargs.get("seed", 0))
        self.submitted = None
        self.tool_tokens = 0
        self.calls = 0
        return None

    def _count(self, text: str) -> str:
        self.calls += 1
        if TOKENIZER is not None:
            self.tool_tokens += len(TOKENIZER(text)["input_ids"])
        return text

    def list_services(self) -> str:
        return self._count("\n".join(f"{s} status=running restarts=0 ready=true" for s in SERVICES))

    def logs(self, service: str) -> str:
        if service not in SERVICES:
            return f"error: unknown service {service!r}"
        rng = random.Random(f"pad-{service}-{self.seed}")
        text = pad_to(
            fake_logs(service, self.seed),
            TOOL_TOKENS,
            lambda i: rng.choice(LOG_TEMPLATES).format(
                ts=f"2026-11-16T09:{i // 60 % 60:02d}:{i % 60:02d}.000Z",
                svc=service,
                rid=f"{rng.getrandbits(48):012x}",
                ms=rng.randint(3, 900),
            ),
        )
        return self._count(text)

    def metrics(self, service: str) -> str:
        if service not in SERVICES:
            return f"error: unknown service {service!r}"
        rng = random.Random(f"m-{service}-{self.seed}")
        err = 0.42 if service in (ROOT_CAUSE, "orders") else rng.random() * 0.01
        text = json.dumps(
            {
                "service": service,
                "error_rate": round(err, 3),
                "p95_ms": rng.randint(20, 9000 if service == ROOT_CAUSE else 300),
                "rps": rng.randint(5, 80),
            }
        )
        text = pad_to(
            text + "\nper-minute history:",
            TOOL_TOKENS,
            lambda i: (
                f"10:{i % 60:02d} err={rng.random() * err:.3f} "
                f"p95_ms={rng.randint(20, 400)} rps={rng.randint(5, 80)}"
            ),
        )
        return self._count(text)

    def submit_diagnosis(self, component: str, category: str) -> str:
        if self.submitted is None:
            self.submitted = (component.strip().lower(), category.strip().lower())
        return "Diagnosis recorded. The episode is over; do not call more tools."

    def get_reward(self) -> float:
        ROLLOUTS.append({"tool_tokens": self.tool_tokens, "tool_calls": self.calls})
        if self.submitted is None:
            return 0.0
        return 1.0 if self.submitted[0] == ROOT_CAUSE else 0.1


PROMPT = (
    "You are the on-call engineer. Users report that checkout fails with 502 errors. "
    "Investigate with the tools. Before deciding, read the logs AND the metrics of every service "
    "listed by list_services, one tool call at a time. Only then call submit_diagnosis once."
)


def model_kwargs(name: str) -> dict:
    kwargs = {}
    if "Instruct-2507" not in name:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    return kwargs


def run_grpo(args) -> dict:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    global TOKENIZER, TOOL_TOKENS
    TOKENIZER = AutoTokenizer.from_pretrained(args.model)
    TOOL_TOKENS = args.tool_tokens
    probe_env = ProbeEnv()
    names = ("list_services", "logs", "metrics", "submit_diagnosis")
    tools = [getattr(probe_env, n) for n in names]
    prompt_tokens = len(
        TOKENIZER.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **model_kwargs(args.model).get("chat_template_kwargs", {}),
        )["input_ids"]
    )
    records = []
    seen = [0]

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
                    "peak_reserved_gb_gpu0": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                    "rollouts": ROLLOUTS[seen[0] :],
                }
            )
            seen[0] = len(ROLLOUTS)

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
        max_completion_length=args.max_len - prompt_tokens - 16,
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
        "prompt_tokens": prompt_tokens,
        "steps": records,
        "nonfinite_steps": unstable,
    }


def run_memory(args) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map={"": 0}
    )
    model.train()
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
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--vllm-port", type=int, default=8000)
    p.add_argument("--tool-tokens", type=int, default=0)
    p.add_argument("--tag", default="")
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
    except Exception:
        import traceback

        result["ok"] = False
        result["error"] = traceback.format_exc()[-3000:]
    tag = args.model.split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    with open(os.path.join(args.out, f"{args.mode}_{tag}.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, default=str)[:2000])


if __name__ == "__main__":
    main()
