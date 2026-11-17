from oncallgym import episode as ep_mod
from oncallgym.episode import Episode
from oncallgym.loop import NUDGE, AgentLoop, parse_reply


def test_parse_reply_hermes():
    r = parse_reply(
        '<think>\n\n</think>\n\nChecking.\n<tool_call>\n{"name": "logs", '
        '"arguments": {"service": "edge"}}\n</tool_call>'
    )
    assert r["content"] == "Checking."
    assert r["tool_calls"] == [{"name": "logs", "arguments": {"service": "edge"}}]
    bad = parse_reply("<tool_call>{not json}</tool_call>")
    assert bad["tool_calls"][0]["name"] is None and "invalid" in bad["tool_calls"][0]["error"]
    assert parse_reply("All done.")["tool_calls"] == []


def _fake_episode(turn_limit=3):
    e = Episode.__new__(Episode)
    e.turns, e.turn_limit, e.termination, e.trajectory, e.started_at = 0, turn_limit, None, [], None
    e.count_tokens = ep_mod.approx_tokens
    e.tokens_used = 0
    e.tools = type("T", (), {"topology": lambda self: "edge :1"})()
    e.start = lambda: [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    return e


def test_text_reply_gets_nudge_and_episode_continues():
    loop = AgentLoop(_fake_episode(turn_limit=3))
    loop.start()
    out = loop.step("I think the edge is broken.")
    assert out == [{"role": "user", "content": NUDGE}] and not loop.done
    out = loop.step('<tool_call>{"name": "topology", "arguments": {}}</tool_call>')
    assert out[0]["role"] == "tool" and "edge" in out[0]["content"] and not loop.done
    loop.step("still thinking")
    assert loop.done and loop.ep.termination == "turn_limit"


def test_invalid_call_costs_a_turn_and_explains_format():
    loop = AgentLoop(_fake_episode())
    loop.start()
    out = loop.step("<tool_call>{oops}</tool_call>")
    assert "invalid tool call" in out[0]["content"] and "<tool_call>" in out[0]["content"]
    assert loop.ep.turns == 1
