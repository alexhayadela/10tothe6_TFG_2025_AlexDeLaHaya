import time
from collections import deque


class RateLimitState:
    def __init__(
        self,
        tpm_limit: int = 8_000,
        rpm_limit: int = 30,
    ):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit

        self.token_events = deque()   # (timestamp, tokens)
        self.request_events = deque() # timestamps only

    def _prune(self, now: float):
        """Remove events older than 60s"""
        while self.token_events and now - self.token_events[0][0] > 60:
            self.token_events.popleft()

        while self.request_events and now - self.request_events[0] > 60:
            self.request_events.popleft()

    def can_make_request(self, estimated_tokens: int) -> float:
        """
        Returns:
          0.0 if request is allowed now
          sleep_seconds if we must wait
        """
        now = time.time()
        self._prune(now)

        used_tokens = sum(t for _, t in self.token_events)
        used_requests = len(self.request_events)

        if used_requests + 1 > self.rpm_limit:
            return 60 - (now - self.request_events[0])

        if used_tokens + estimated_tokens > self.tpm_limit:
            return 60 - (now - self.token_events[0][0])

        return 0.0

    def record(self, tokens_used: int):
        now = time.time()
        self.token_events.append((now, tokens_used))
        self.request_events.append(now)
