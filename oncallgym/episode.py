import hashlib
import inspect
import json
import os
import re
import time

from . import sandbox
from .faults import CATEGORIES, make_fault
from .faults.base import configure_quietly
from .prompt import system_prompt, task_prompt
from .stack import Stack, http_json
from .verifier import VERIFIER_VERSION, Baseline, Breakdown, verify

LIMITS = {
    "max_output_chars": 2000,
    "context_tokens": 8192,
    "reserve_tokens": 512,
    "model_tokens_per_turn": 160,
    "exec_timeout_s": 10,
    "logs_max_tail": 50,
    "line_max_chars": 240,
}
TERMINATIONS = ("submitted", "turn_limit", "context_exhausted", "no_submission", "error")


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2 - 20
    return text[:half] + "\n...[output truncated]...\n" + text[-half:]


class EpisodeOver(Exception):
    pass


class Tools:
    def __init__(self, ep: "Episode"):
        self._ep = ep

    def topology(self) -> str:
        from northwind.topology import SERVICES

        ports = self._ep.stack.ports
        lines = [
            f"{n} :{ports[n]} {spec[3]} | depends on: {', '.join(spec[2]) or '-'}"
            for n, spec in SERVICES.items()
        ]
        lines.append(f"supervisor :{ports['supervisor']} (process manager, use restart tool)")
        return "\n".join(lines)

    def list_services(self) -> str:
        rows = ["service      state    ready restarts oom rss_mb  not_ready_reason"]
        for s in self._ep.stack.services():
            reason = "; ".join(s["not_ready_reasons"] or [])[:80] if not s["ready"] else ""
            rows.append(
                f"{s['name']:12s} {s['state']:8s} {'yes' if s['ready'] else 'NO ':5s} "
                f"{s['restarts']:<8d} {s['oom_kills']:<3d} {s['rss_mb'] or '-':<7} {reason}"
            )
        return "\n".join(rows)

    def exec(self, command: str) -> str:
        r = sandbox.run(
            self._ep.agent_user,
            command,
            cwd=self._ep.stack.home,
            timeout_s=LIMITS["exec_timeout_s"],
        )
        status = "timed out" if r["timed_out"] else f"exit {r['exit_code']}"
        return f"[{status}]\n{r['output']}"

    def logs(self, service: str, tail: int = 20, grep: str = "") -> str:
        from northwind.topology import SERVICES

        if service not in SERVICES and service != "supervisor":
            return f"unknown service {service!r}"
        tail = max(1, min(int(tail), LIMITS["logs_max_tail"]))
        path = os.path.join(self._ep.stack.home, "logs", f"{service}.log")
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 400_000))
                lines = f.read().decode(errors="replace").splitlines()[1:]
        except OSError:
            return f"no log file for {service}"
        if grep:
            needle = str(grep)[:200].lower()
            lines = [ln for ln in lines if needle in ln.lower()]
        else:
            lines = [ln for ln in lines if "type=access" not in ln or " status=2" not in ln]
        lines = [ln[: LIMITS["line_max_chars"]] for ln in lines[-tail:]]
        return "\n".join(lines) or "(no matching lines)"

    def metrics(self, service: str = "") -> str:
        port = self._ep.stack.ports["telemetry"]
        if service:
            st, body = http_json(port, "GET", f"/services/{service}")
            if st != 200:
                return f"telemetry: {body.get('error', st)}"
            body.pop("service", None)
            return f"{service}: " + json.dumps(body, separators=(",", ":"))
        st, body = http_json(port, "GET", "/summary")
        if st != 200:
            return f"telemetry unavailable ({st})"
        rows = ["service      up ready rps    err%  p95ms  rss_mb log_err"]
        for s in body["services"]:
            err = s.get("error_rate")
            rows.append(
                f"{s['service']:12s} {'y' if s['up'] else 'N':2s} {'y' if s['ready'] else 'N':5s} "
                f"{s.get('rps') if s.get('rps') is not None else '-':<6} "
                f"{round(err * 100, 1) if err is not None else '-':<5} "
                f"{s.get('p95_ms') if s.get('p95_ms') is not None else '-':<6} "
                f"{s.get('rss_mb', '-'):<6} {s.get('log_errors', 0)}"
            )
        return "\n".join(rows)

    def trace(self, request_id: str) -> str:
        port = self._ep.stack.ports["telemetry"]
        if request_id == "recent_errors":
            st, body = http_json(port, "GET", "/traces?errors=1&limit=8")
            rows = [
                f"{t['request_id']} {t['path']} status={t['status']}"
                for t in body.get("traces", [])
            ]
            return "\n".join(rows) or "(no failed requests in recent traces)"
        st, body = http_json(port, "GET", f"/trace/{request_id}")
        if st != 200:
            return body.get("error", f"status {st}")
        lines = [
            f"{h.get('ts', '')[11:23]} {h.get('caller', '?')} -> {h.get('svc')} "
            f"{h.get('method')} {h.get('path')} {h.get('status')} {h.get('dur_ms')}ms"
            for h in body["hops"]
        ]
        for e in body["events"]:
            extra = " ".join(
                f"{k}={v}" for k, v in e.items() if k not in ("ts", "svc", "level", "msg")
            )
            lines.append(f"  {e.get('level')} {e.get('svc')}: {e.get('msg')} {extra}")
        return "\n".join(lines)

    def restart(self, service: str) -> str:
        from northwind.topology import SERVICES

        if service not in SERVICES:
            return f"unknown service {service!r}"
        st, body = self._ep.stack.restart(service)
        if st != 200:
            return f"restart failed: {body}"
        deadline = time.time() + 15
        while time.time() < deadline:
            s = {x["name"]: x for x in self._ep.stack.services()}[service]
            if s["ready"] and s["state"] == "running":
                return f"{service} restarted and ready (pid {s['pid']})"
            time.sleep(0.5)
        reasons = "; ".join(s.get("not_ready_reasons") or [s["state"]])
        return f"{service} restarted but NOT ready after 15s: {reasons}"

    def submit_diagnosis(self, root_cause_category: str, component: str, summary: str) -> str:
        self._ep._diagnosis = {
            "root_cause_category": root_cause_category,
            "component": component,
            "summary": summary,
        }
        self._ep._end("submitted")
        valid = root_cause_category in CATEGORIES
        return "diagnosis submitted" + ("" if valid else " (unknown category)")


TOOL_DOCS = {
    "topology": (
        "Show the services, what each one depends on, and their ports.\n"
        "\n"
        "        Returns:\n"
        "            One line per service: name, port, role, dependencies.\n"
        "        "
    ),
    "list_services": (
        "List every service with its state, readiness, restart count and memory.\n"
        "\n"
        "        Returns:\n"
        "            A table with one row per service.\n"
        "        "
    ),
    "exec": (
        "Run a shell command in the deployment directory (etc/, logs/, data/).\n"
        "\n"
        "        Use it to read and edit config files or call service HTTP APIs with curl.\n"
        "        Output is truncated; commands time out after a few seconds.\n"
        "\n"
        "        Args:\n"
        "            command: The shell command to run.\n"
        "\n"
        "        Returns:\n"
        "            Exit code and combined stdout/stderr.\n"
        "        "
    ),
    "logs": (
        "Show recent log lines of one service.\n"
        "\n"
        "        Args:\n"
        '            service: Service name, e.g. "orders".\n'
        "            tail: Number of most recent matching lines to show (max 50).\n"
        "            grep: Optional text; only lines containing it (case-insensitive) are shown.\n"
        "\n"
        "        Returns:\n"
        "            Matching log lines, oldest first.\n"
        "        "
    ),
    "metrics": (
        "Show metrics from telemetry over the last 30 seconds.\n"
        "\n"
        "        Args:\n"
        "            service: Optional service name. Without it, a one-line summary per service.\n"
        "\n"
        "        Returns:\n"
        "            Error rate, p95 latency, request rate, memory, and where relevant circuit\n"
        "            breakers, connection pools, queue depths / dead letters, cache usage.\n"
        "        "
    ),
    "trace": (
        "Follow one request through every service it touched.\n"
        "\n"
        "        Args:\n"
        '            request_id: A request id (req=... in logs). Use "recent_errors" to '
        "list recent\n"
        "                failed requests.\n"
        "\n"
        "        Returns:\n"
        "            The hops of the request with status and duration, plus "
        "warnings/errors logged\n"
        "            for it.\n"
        "        "
    ),
    "restart": (
        "Restart one service (it re-reads its config and env file).\n"
        "\n"
        "        Args:\n"
        '            service: Service name, e.g. "orders".\n'
        "\n"
        "        Returns:\n"
        "            Whether the service came back ready.\n"
        "        "
    ),
    "submit_diagnosis": (
        "Submit your diagnosis. This ends the episode; call it once, after fixing.\n"
        "\n"
        "        Args:\n"
        "            root_cause_category: One of the root cause categories listed in the "
        "instructions.\n"
        "            component: The service where the root cause lives.\n"
        "            summary: One or two sentences on what was wrong and what you changed.\n"
        "\n"
        "        Returns:\n"
        "            Confirmation that the episode ended.\n"
        "        "
    ),
}

TOOL_NAMES = [n for n, f in inspect.getmembers(Tools, inspect.isfunction) if not n.startswith("_")]


def tool_schemas_fingerprint() -> str:
    h = hashlib.sha256()
    for n in TOOL_NAMES:
        f = getattr(Tools, n)
        h.update(n.encode())
        h.update(str(inspect.signature(f)).encode())
        h.update(TOOL_DOCS.get(n, "").encode())
    return h.hexdigest()[:16]


def fingerprint() -> str:
    h = hashlib.sha256()
    h.update(system_prompt().encode())
    h.update(tool_schemas_fingerprint().encode())
    h.update(VERIFIER_VERSION.encode())
    h.update(json.dumps(LIMITS, sort_keys=True).encode())
    return h.hexdigest()[:16]


class Episode:
    def __init__(
        self,
        family: str,
        variant: int = 0,
        seed: int | None = None,
        count_tokens=None,
        root: str | None = None,
    ):
        self.fault = make_fault(family, variant)
        self.seed = variant if seed is None else seed
        self.count_tokens = approx_tokens
        self.real_count_tokens = count_tokens
        self.stack = Stack(seed=self.seed, root=root)
        self.tools = Tools(self)
        self.turn_limit = self.fault.turn_limit
        self.turns = 0
        self.tokens_used = 0
        self.termination: str | None = None
        self._diagnosis = None
        self.baseline: Baseline | None = None
        self.breakdown: Breakdown | None = None
        self.trajectory: list[dict] = []
        self.started_at = None

    def start(self) -> list[dict]:
        self.started_at = time.time()
        self.stack.create()
        self.agent_user = sandbox.agent_user(self.stack.block)
        sandbox.ensure_agent_user(self.agent_user)
        sandbox.wipe(self.agent_user)
        configure_quietly(self.stack, self.fault)
        self.stack.start(timeout_s=45, require_ready=False)
        time.sleep(2)
        self.fault.inject(self.stack)
        sandbox.lock_down(self.stack.home, self.agent_user)
        self.baseline = Baseline.capture(self.stack)
        time.sleep(self.fault.settle_s)
        msgs = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": task_prompt(self.fault.alert(), self.turn_limit)},
        ]
        self.tokens_used = sum(self.count_tokens(m["content"]) for m in msgs) + self.count_tokens(
            json.dumps(tool_schemas())
        )
        self._log("start", variant=self.fault.variant_id, messages=msgs, fingerprint=fingerprint())
        return msgs

    @property
    def done(self) -> bool:
        return self.termination is not None

    def note_model_turn(self, text: str):
        self._sees_model_text = True
        self.tokens_used += self.count_tokens(text)
        self._log("model", text=text)

    def nudge(self) -> str:
        from .loop import NUDGE

        if self.done:
            return f"episode is over ({self.termination})"
        self.turns += 1
        return self._deliver("__nudge__", {}, NUDGE)

    def call_tool(self, name: str, args: dict | None = None) -> str:
        args = args or {}
        if self.done:
            return f"episode is over ({self.termination})"
        self.turns += 1
        if name == "__invalid__":
            out = (
                f"{args.get('error', 'invalid tool call')}. Write calls as <tool_call>"
                '{"name": ..., "arguments": {...}}</tool_call>.'
            )
        elif name not in TOOL_NAMES:
            out = f"unknown tool {name!r}; available: {', '.join(TOOL_NAMES)}"
        else:
            try:
                out = getattr(self.tools, name)(**args)
            except TypeError as e:
                out = f"bad arguments for {name}: {e}"
            except Exception as e:
                out = f"tool error: {type(e).__name__}: {e}"[:300]
        return self._deliver(name, args, out)

    def _deliver(self, name: str, args: dict, out) -> str:
        out = _cap(str(out), LIMITS["max_output_chars"])
        allowance = (
            0 if getattr(self, "_sees_model_text", False) else LIMITS["model_tokens_per_turn"]
        )
        cost = self.count_tokens(out) + allowance
        if not self.done and (
            self.tokens_used + cost + LIMITS["reserve_tokens"] > LIMITS["context_tokens"]
        ):
            self._log("tool", name=name, args=args, output=None, dropped_tokens=cost)
            self._end("context_exhausted")
            return "context budget exhausted; the episode is over."
        self.tokens_used += cost
        self._log(
            "tool" if name != "__nudge__" else "nudge",
            name=name,
            args=args,
            output=out,
            tokens=cost,
        )
        if not self.done and self.turns >= self.turn_limit:
            self._end("turn_limit")
            out += "\n[turn limit reached; the episode is over]"
        return out

    def _end(self, reason: str):
        if self.termination is None:
            assert reason in TERMINATIONS
            self.termination = reason

    def finish(self) -> Breakdown:
        if self.breakdown is not None:
            return self.breakdown
        if self.termination is None:
            self._end("turn_limit" if self.turns >= self.turn_limit else "no_submission")
        try:
            b = verify(
                self.stack,
                self.fault.ground_truth,
                self._diagnosis,
                self.baseline,
                termination=self.termination,
                fixed_check=self.fault.root_cause_fixed,
            )
        except Exception as e:
            b = Breakdown(
                termination="error", details={"verifier_error": f"{type(e).__name__}: {e}"[:300]}
            )
            b.finalize()
        b.details["turns"] = self.turns
        b.details["tokens_used"] = self.tokens_used
        b.details["diagnosis"] = self._diagnosis
        self.breakdown = b
        self._log("finish", breakdown=b.to_dict(), ground_truth=self.fault.ground_truth.__dict__)
        return b

    def close(self):
        if getattr(self, "agent_user", None):
            try:
                sandbox.wipe(self.agent_user)
            except Exception:
                pass
        self.stack.cleanup()

    def _log(self, kind: str, **fields):
        self.trajectory.append(
            {"t": round(time.time() - (self.started_at or time.time()), 2), "kind": kind, **fields}
        )


_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean"}


def tool_schemas() -> list[dict]:
    out = []
    for name in TOOL_NAMES:
        fn = getattr(Tools, name)
        doc = inspect.cleandoc(TOOL_DOCS.get(name, ""))
        desc = doc.split("\n\n")[0].replace("\n", " ").strip()
        arg_docs, current = {}, None
        in_args = False
        for line in doc.splitlines():
            if line.strip() == "Args:":
                in_args = True
                continue
            if in_args:
                if line and not line.startswith(" "):
                    break
                m = re.match(r"\s{4}(\w+): (.*)", line)
                if m:
                    current = m.group(1)
                    arg_docs[current] = m.group(2).strip()
                elif current and line.strip():
                    arg_docs[current] += " " + line.strip()
        props, required = {}, []
        for p in list(inspect.signature(fn).parameters.values())[1:]:
            props[p.name] = {
                "type": _PY_TO_JSON[p.annotation],
                "description": arg_docs.get(p.name, ""),
            }
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required},
                },
            }
        )
    return out
