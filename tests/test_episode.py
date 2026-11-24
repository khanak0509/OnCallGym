import platform

import pytest

from oncallgym import episode as ep_mod
from oncallgym.episode import LIMITS, TOOL_NAMES, Episode, fingerprint

linux_only = pytest.mark.skipif(platform.system() != "Linux", reason="needs a Linux stack")


def test_tool_names_and_fingerprint_stable():
    assert TOOL_NAMES == sorted(
        [
            "exec",
            "list_services",
            "logs",
            "metrics",
            "restart",
            "submit_diagnosis",
            "topology",
            "trace",
        ]
    )
    assert fingerprint() == fingerprint()


@pytest.fixture(scope="module")
def episode(tmp_path_factory):
    e = Episode("jwt_key_rotation_mismatch", variant=0, root=str(tmp_path_factory.mktemp("ep")))
    e.start()
    yield e
    e.close()


@pytest.mark.stack
@linux_only
def test_agent_cannot_write_data_or_read_secrets(episode):
    out = episode.call_tool("exec", {"command": "echo x >> data/outbox/emails.jsonl; echo rc=$?"})
    assert "Permission denied" in out and "rc=0" not in out
    out = episode.call_tool("exec", {"command": "sqlite3 data/orders.db 'DELETE FROM orders'"})
    assert "readonly" in out.lower() or "unable" in out.lower() or "denied" in out.lower()
    out = episode.call_tool("exec", {"command": f"cat {episode.stack.private}/verifier.json"})
    assert "Permission denied" in out
    out = episode.call_tool("exec", {"command": "kill -9 $(cat run/supervisor.pid); echo rc=$?"})
    assert "rc=1" in out or "not permitted" in out


@pytest.mark.stack
@linux_only
def test_agent_can_edit_config(episode):
    out = episode.call_tool(
        "exec", {"command": "sed -i 's/level: info/level: info/' etc/orders.yaml && echo edited"}
    )
    assert "edited" in out


@pytest.mark.stack
@linux_only
def test_tools_return_compact_output(episode):
    for name, args in [
        ("topology", {}),
        ("list_services", {}),
        ("metrics", {}),
        ("logs", {"service": "orders", "tail": 50}),
        ("trace", {"request_id": "recent_errors"}),
    ]:
        out = episode.call_tool(name, args)
        assert len(out) <= LIMITS["max_output_chars"] + 40, name


def test_context_exhausted_ends_cleanly(monkeypatch):
    e = Episode.__new__(Episode)
    e.turns, e.turn_limit, e.termination, e.trajectory, e.started_at = 0, 50, None, [], None
    e.count_tokens = ep_mod.approx_tokens
    e.tokens_used = LIMITS["context_tokens"] - LIMITS["reserve_tokens"] - 10
    e.tools = type("T", (), {"topology": lambda self: "x" * 1000})()
    out = e.call_tool("topology")
    assert e.termination == "context_exhausted" and "exhausted" in out
    assert e.call_tool("topology").startswith("episode is over")


def test_turn_limit_and_unknown_tools_count(monkeypatch):
    e = Episode.__new__(Episode)
    e.turns, e.turn_limit, e.termination, e.trajectory, e.started_at = 0, 2, None, [], None
    e.count_tokens = ep_mod.approx_tokens
    e.tokens_used = 0
    e.tools = type("T", (), {"topology": lambda self: "ok"})()
    e.call_tool("nope")
    out = e.call_tool("topology")
    assert e.termination == "turn_limit" and "turn limit" in out
