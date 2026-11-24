import os
import sys
import threading
import time

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}


def _fmt_value(v) -> str:
    s = str(v)
    if s == "" or any(c in s for c in ' ="\n\t'):
        s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'"{s}"'
    return s


def utc_ts(t: float | None = None) -> str:
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t * 1000) % 1000:03d}Z"


def format_line(level: str, svc: str, msg: str, fields: dict) -> str:
    parts = [f"ts={utc_ts()}", f"level={level}", f"svc={svc}"]
    req = fields.pop("req", None)
    if req:
        parts.append(f"req={req}")
    parts.append(f"msg={_fmt_value(msg)}")
    for k, v in fields.items():
        if v is not None:
            parts.append(f"{k}={_fmt_value(v)}")
    return " ".join(parts)


def parse_line(line: str) -> dict:
    out = {}
    i, n = 0, len(line)
    while i < n:
        while i < n and line[i] == " ":
            i += 1
        j = line.find("=", i)
        if j < 0:
            break
        key = line[i:j]
        i = j + 1
        if i < n and line[i] == '"':
            i += 1
            buf = []
            while i < n and line[i] != '"':
                if line[i] == "\\" and i + 1 < n:
                    nxt = line[i + 1]
                    buf.append("\n" if nxt == "n" else nxt)
                    i += 2
                else:
                    buf.append(line[i])
                    i += 1
            i += 1
            out[key] = "".join(buf)
        else:
            j = line.find(" ", i)
            j = n if j < 0 else j
            out[key] = line[i:j]
            i = j
    return out


class Logger:
    def __init__(self, svc: str, path: str | None, level: str = "info"):
        self.svc = svc
        self.path = path
        self.min_level = LEVELS.get(level, 20)
        self._lock = threading.Lock()
        self._fh = None
        self._ino = None
        self._last_check = 0.0
        self.write_errors = 0

    def _open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._ino = os.fstat(self._fh.fileno()).st_ino

    def _maybe_reopen(self):
        now = time.monotonic()
        if self._fh is not None and now - self._last_check < 1.0:
            return
        self._last_check = now
        try:
            ino = os.stat(self.path).st_ino
        except FileNotFoundError:
            ino = None
        if self._fh is None or ino != self._ino:
            if self._fh is not None:
                self._fh.close()
            self._open()

    def log(self, level: str, msg: str, **fields):
        if LEVELS[level] < self.min_level:
            return
        line = format_line(level, self.svc, msg, fields)
        if self.path is None:
            print(line, file=sys.stderr)
            return
        with self._lock:
            try:
                self._maybe_reopen()
                self._fh.write(line + "\n")
            except OSError:
                self.write_errors += 1
                self._fh = None

    def debug(self, msg, **f):
        self.log("debug", msg, **f)

    def info(self, msg, **f):
        self.log("info", msg, **f)

    def warn(self, msg, **f):
        self.log("warn", msg, **f)

    def error(self, msg, **f):
        self.log("error", msg, **f)
