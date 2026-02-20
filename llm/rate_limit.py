import time
from collections import deque

class RateLimitState:
    def __init__(self, tpm_limit: int = 8000, rpm_limit: int = 30):
        self.tpm_limit = tpm_limit
        self.rpm_limit = rpm_limit
        self.token_events = deque()   # (timestamp, tokens)
        self.request_events = deque() # timestamps only

    def _prune(self, now: float) -> None:
        """Remove events older than 60 seconds."""
        while self.token_events and now - self.token_events[0][0] > 60:
            self.token_events.popleft()
        while self.request_events and now - self.request_events[0] > 60:
            self.request_events.popleft()

    def wait_for_slot(self, estimated_tokens: int) -> None:
        """
        Checks if a request is allowed.
        Sleeps if necessary until the window allows the request.
        """
        while True:
            now = time.time()
            self._prune(now)

            used_tokens = sum(t for _, t in self.token_events)
            used_requests = len(self.request_events)

            sleep_seconds = 0.0

            # Check RPM
            if used_requests + 1 > self.rpm_limit and self.request_events:
                sleep_seconds = 60 - (now - self.request_events[0])

            # Check TPM
            if used_tokens + estimated_tokens > self.tpm_limit and self.request_events:
                t_sleep = 60 - (now - self.token_events[0][0])
                sleep_seconds = max(sleep_seconds, t_sleep)

            if sleep_seconds > 0:
                print(f"⏳ Rate limit exceeded, sleeping {sleep_seconds:.2f}s")
                time.sleep(sleep_seconds)
            else:
                break  # ok to proceed

    def record(self, tokens_used: int) -> None:
        """Records tokens used."""
        now = time.time()
        self.token_events.append((now, tokens_used))
        self.request_events.append(now)

