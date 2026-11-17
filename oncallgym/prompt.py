"""System prompt and task prompt. Shared by the eval (verifiers) and training (TRL) paths."""

from .faults import CATEGORIES

SYSTEM_PROMPT = """You are the on-call engineer for Northwind Commerce, an online shop made of \
about fifteen microservices. An alert fired and customers are affected. Find the root cause, \
fix it so the system recovers and stays recovered after restarts, then submit your diagnosis.

Rules:
- Use the tools. Shell commands run in the deployment directory: service configs are \
etc/<service>.yaml, environment files etc/env/<service>.env, feature flags \
etc/feature-flags.yaml, logs logs/<service>.log, data in data/. You may edit config \
files; you may not change code, data, the external \
payment provider, the traffic generator or telemetry.
- Fix the cause, not the symptom. Restarting alone does not fix configuration or data problems.
- Do not delete or rewrite business data, messages or logs.
- You have a limited number of tool calls. submit_diagnosis ends the episode and can be \
called once.

Root cause categories:
{categories}
Component is the name of the one service whose configuration, data or code holds the root \
cause (as listed by topology), not the service where the symptom shows."""

CATEGORY_HELP = {
    "config": "a wrong value in a service's configuration (paths, ports, routes, topics, env "
    "vars, rate limits, outdated config)",
    "resilience_config": "timeouts, retries, circuit breakers or connection pools tuned wrongly",
    "resource_exhaustion": "memory, disk or cache capacity running out",
    "auth": "signing keys, tokens or credentials not matching",
    "bad_deploy": "a bad build or version flag in a deployed service",
    "schema": "a database schema that does not match what the code needs",
    "performance": "slow database queries (e.g. a missing index) or lock contention",
    "data": "bad data or messages, or message queue settings (dead letters, redelivery)",
    "dependency_down": "a service that others depend on is unavailable",
}


def system_prompt() -> str:
    cats = "\n".join(f"- {c}: {CATEGORY_HELP[c]}" for c in CATEGORIES)
    return SYSTEM_PROMPT.format(categories=cats)


def task_prompt(alert: str, turn_limit: int) -> str:
    return f"ALERT: {alert}\nYou have {turn_limit} tool calls. Start investigating."
