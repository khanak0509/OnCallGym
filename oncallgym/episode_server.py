"""Expose one Episode over local HTTP so tool code in another process (verifiers runs each
toolset as its own process) calls the same Episode object. No logic lives here.

  POST /call    {"name": ..., "args": {...}}  -> {"output": str, "done": bool}
  POST /model   {"text": ...}                 -> {}   (count a model turn in the budget)
  POST /finish                                -> breakdown dict
  GET  /state                                 -> {"done", "termination", "turns", "tokens_used"}
"""

import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .episode import Episode


class EpisodeServer:
    def __init__(self, episode: Episode):
        self.ep = episode
        # every request must carry this token (other local processes, e.g. something the
        # agent started, cannot drive the episode)
        self.token = os.urandom(16).hex()
        self.lock = threading.Lock()  # tool calls of one episode are sequential
        server = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj, default=str).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorized(self):
                return hmac.compare_digest(self.headers.get("X-Episode-Token", ""), server.token)

            def do_GET(self):
                if not self._authorized():
                    return self._send(403, {"error": "forbidden"})
                if self.path == "/state":
                    e = server.ep
                    return self._send(
                        200,
                        {
                            "done": e.done,
                            "termination": e.termination,
                            "turns": e.turns,
                            "tokens_used": e.tokens_used,
                        },
                    )
                self._send(404, {"error": "not found"})

            def do_POST(self):
                if not self._authorized():
                    return self._send(403, {"error": "forbidden"})
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                with server.lock:
                    if self.path == "/call":
                        out = server.ep.call_tool(body["name"], body.get("args") or {})
                        return self._send(200, {"output": out, "done": server.ep.done})
                    if self.path == "/model":
                        server.ep.note_model_turn(body.get("text", ""))
                        return self._send(200, {})
                    if self.path == "/finish":
                        return self._send(200, server.ep.finish().to_dict())
                self._send(404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
