import threading

DEFAULT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _labels(labels: dict) -> str:
    if not labels:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class Registry:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[tuple, float] = {}
        self._gauges: dict[tuple, float] = {}
        self._gauge_fns: dict[tuple, callable] = {}
        self._hists: dict[tuple, list] = {}
        self._buckets: dict[str, tuple] = {}

    @staticmethod
    def _key(name, labels):
        return (name, tuple(sorted((labels or {}).items())))

    def inc(self, name: str, value: float = 1.0, **labels):
        k = self._key(name, labels)
        with self._lock:
            self._counters[k] = self._counters.get(k, 0.0) + value

    def set(self, name: str, value: float, **labels):
        with self._lock:
            self._gauges[self._key(name, labels)] = float(value)

    def gauge_fn(self, name: str, fn, **labels):
        self._gauge_fns[self._key(name, labels)] = fn

    def observe(self, name: str, value_ms: float, buckets=DEFAULT_BUCKETS, **labels):
        k = self._key(name, labels)
        with self._lock:
            self._buckets.setdefault(name, buckets)
            h = self._hists.get(k)
            if h is None:
                h = [0] * (len(buckets) + 1) + [0.0, 0]
                self._hists[k] = h
            for i, b in enumerate(buckets):
                if value_ms <= b:
                    h[i] += 1
            h[len(buckets)] += 1
            h[-2] += value_ms
            h[-1] += 1

    def counter_value(self, name: str, **labels) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def render(self) -> str:
        lines = []
        with self._lock:
            for (name, labels), v in sorted(self._counters.items()):
                lines.append(f"{name}{_labels(dict(labels))} {v:g}")
            for (name, labels), v in sorted(self._gauges.items()):
                lines.append(f"{name}{_labels(dict(labels))} {v:g}")
            for (name, labels), h in sorted(self._hists.items()):
                buckets = self._buckets[name]
                lab = dict(labels)
                for i, b in enumerate(buckets):
                    lines.append(f"{name}_bucket{_labels({**lab, 'le': b})} {h[i]}")
                lines.append(f"{name}_bucket{_labels({**lab, 'le': '+Inf'})} {h[len(buckets)]}")
                lines.append(f"{name}_sum{_labels(lab)} {h[-2]:g}")
                lines.append(f"{name}_count{_labels(lab)} {h[-1]}")
        for (name, labels), fn in sorted(self._gauge_fns.items(), key=lambda kv: kv[0]):
            try:
                v = float(fn())
            except Exception:
                continue
            lines.append(f"{name}{_labels(dict(labels))} {v:g}")
        return "\n".join(lines) + "\n"


def parse_text(text: str) -> list[tuple[str, dict, float]]:
    out = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            head, val = line.rsplit(" ", 1)
            value = float(val)
        except ValueError:
            continue
        labels = {}
        if "{" in head:
            name, rest = head.split("{", 1)
            for part in rest.rstrip("}").split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    labels[k] = v.strip('"')
        else:
            name = head
        out.append((name, labels, value))
    return out
