import json
import re

NUDGE = (
    "You replied without calling a tool. Keep investigating with the tools, or call "
    "submit_diagnosis once you have fixed the problem."
)
_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def parse_reply(text: str) -> dict:
    text = _THINK.sub("", text or "")
    calls = []
    for raw in _CALL.findall(text):
        try:
            obj = json.loads(raw)
            name = obj.get("name")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError("need a name and an arguments object")
            calls.append({"name": name, "arguments": args})
        except (ValueError, AttributeError) as e:
            calls.append({"name": None, "arguments": {}, "error": f"invalid tool call: {e}"})
    content = _CALL.sub("", text).strip()
    return {"content": content, "tool_calls": calls}


def assistant_message(parsed: dict) -> dict:
    msg = {"role": "assistant", "content": parsed["content"]}
    good = [c for c in parsed["tool_calls"] if c.get("name")]
    if good:
        msg["tool_calls"] = [
            {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in good
        ]
    return msg


class AgentLoop:
    def __init__(self, episode):
        self.ep = episode
        self.messages: list[dict] = []

    @property
    def done(self) -> bool:
        return self.ep.done

    def start(self) -> list[dict]:
        self.messages = self.ep.start()
        return self.messages

    def step(self, reply_text: str) -> list[dict]:
        parsed = parse_reply(reply_text)
        self.ep.note_model_turn(reply_text)
        self.messages.append(assistant_message(parsed))
        out = []
        if not parsed["tool_calls"]:
            text = self.ep.nudge()
            out.append({"role": "user", "content": text})
        for call in parsed["tool_calls"]:
            if self.ep.done:
                break
            if call.get("name") is None:
                result = self.ep.call_tool("__invalid__", {"error": call["error"]})
            else:
                result = self.ep.call_tool(call["name"], call["arguments"])
            out.append({"role": "tool", "name": call.get("name") or "invalid", "content": result})
        self.messages.extend(out)
        return out
