import time
import pytest
from llm.rate_limit import RateLimitState


def test_record_increments_deques():
    rl = RateLimitState(tpm_limit=10000, rpm_limit=100)
    rl.record(500)
    assert len(rl.token_events) == 1
    assert len(rl.request_events) == 1


def test_prune_removes_old_events():
    rl = RateLimitState()
    old_ts = time.time() - 120  # 2 minutes ago
    rl.token_events.append((old_ts, 100))
    rl.request_events.append(old_ts)
    rl._prune(time.time())
    assert len(rl.token_events) == 0
    assert len(rl.request_events) == 0


def test_prune_keeps_recent_events():
    rl = RateLimitState()
    rl.record(100)
    rl._prune(time.time())
    assert len(rl.token_events) == 1


def test_wait_for_slot_no_sleep_under_limits():
    rl = RateLimitState(tpm_limit=10000, rpm_limit=100)
    start = time.time()
    rl.wait_for_slot(estimated_tokens=100)
    elapsed = time.time() - start
    assert elapsed < 0.5
