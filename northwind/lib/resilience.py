import random
import threading
import time

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"
STATE_CODE = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}


class CircuitBreaker:
    def __init__(self, failure_threshold=5, window_s=10.0, reset_timeout_s=5.0, half_open_max=1):
        self.failure_threshold = int(failure_threshold)
        self.window_s = float(window_s)
        self.reset_timeout_s = float(reset_timeout_s)
        self.half_open_max = int(half_open_max)
        self.state = CLOSED
        self._failures: list[float] = []
        self._opened_at = 0.0
        self._trials = 0
        self._lock = threading.Lock()
        self.on_change = None

    def _set(self, state):
        if state != self.state:
            old, self.state = self.state, state
            if self.on_change:
                self.on_change(old, state)

    def allow(self) -> bool:
        with self._lock:
            if self.state == OPEN:
                if time.monotonic() - self._opened_at >= self.reset_timeout_s:
                    self._set(HALF_OPEN)
                    self._trials = 0
                else:
                    return False
            if self.state == HALF_OPEN:
                if self._trials >= self.half_open_max:
                    return False
                self._trials += 1
            return True

    def release_trial(self):
        with self._lock:
            if self.state == HALF_OPEN and self._trials > 0:
                self._trials -= 1

    def success(self):
        with self._lock:
            if self.state == HALF_OPEN:
                self._failures.clear()
                self._set(CLOSED)

    def failure(self):
        with self._lock:
            now = time.monotonic()
            if self.state == HALF_OPEN:
                self._opened_at = now
                self._set(OPEN)
                return
            self._failures = [t for t in self._failures if now - t < self.window_s]
            self._failures.append(now)
            if len(self._failures) >= self.failure_threshold:
                self._opened_at = now
                self._set(OPEN)


def backoff_delays(retries: int, base_ms: float, max_ms: float, jitter: bool = True):
    for i in range(retries):
        d = min(max_ms, base_ms * (2**i))
        if jitter:
            d = random.uniform(d / 2, d)
        yield d / 1000.0


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: float):
        self.rate = float(rate_per_s)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.t = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
            self.t = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False
