import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT = [
    ("list_services", {}),
    ("logs", {"service": "orders", "tail": 5, "grep": "token"}),
    (
        "submit_diagnosis",
        {"root_cause_category": "auth", "component": "orders", "summary": "scripted"},
    ),
]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        d = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(d)))
        self.end_headers()
        self.wfile.write(d)

    def do_GET(self):
        self._send({"data": [{"id": "model", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n_tool = sum(1 for m in body.get("messages", []) if m.get("role") == "tool")
        msg = {"role": "assistant", "content": "done"}
        if n_tool < len(SCRIPT):
            name, args = SCRIPT[n_tool]
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c" + uuid.uuid4().hex[:8],
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        if body.get("stream"):
            delta = dict(msg)
            if delta.get("tool_calls"):
                delta["tool_calls"] = [dict(tc, index=0) for tc in delta["tool_calls"]]
            chunks = [
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            ]
            data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
            d = data.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(d)))
            self.end_headers()
            self.wfile.write(d)
            return
        self._send(
            {
                "id": "x",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "message": msg,
                        "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        )


if __name__ == "__main__":
    import sys

    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
