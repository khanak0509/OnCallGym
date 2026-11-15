# OnCallGym

An RL environment where an agent works as the on-call engineer for a simulated microservice backend. Each episode starts the system, injects a production fault, gives the agent ops tools, and scores the fix with code (no LLM judge).

Work in progress.

## Setup

```
uv sync
make test
```

For local development, run everything inside the container: `make shell`.
