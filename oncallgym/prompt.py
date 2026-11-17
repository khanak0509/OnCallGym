"""System prompt and task prompt. Shared by the eval (verifiers) and training (TRL) paths."""

from .faults import CATEGORIES

SYSTEM_PROMPT = """You are the on-call engineer for Northwind Commerce, an online shop made of \
about fifteen microservices. An alert fired and customers are affected. Find the root cause, \
fix it so the system recovers and stays recovered after restarts, then submit your diagnosis.

Rules:
- Use the tools. Shell commands run in the deployment directory (configs in etc/, logs in \
logs/, data in data/). You may edit config files; you may not change code, data, the external \
payment provider, the traffic generator or telemetry.
- Fix the cause, not the symptom. Restarting alone does not fix configuration or data problems.
- Do not delete or rewrite business data, messages or logs.
- You have a limited number of tool calls. submit_diagnosis ends the episode and can be \
called once.

Root cause categories: {categories}.
Component is the name of the one service where the root cause lives (as listed by topology)."""


def system_prompt() -> str:
    return SYSTEM_PROMPT.format(categories=", ".join(CATEGORIES))


def task_prompt(alert: str, turn_limit: int) -> str:
    return f"ALERT: {alert}\nYou have {turn_limit} tool calls. Start investigating."
