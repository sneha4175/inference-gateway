"""Rate limiter: burst is capped, and the bucket refills over time."""

from app.gateway.rate_limiter import RateLimiter


def test_burst_capacity_is_enforced():
    # 60/min, burst of 3 -> first 3 calls allowed, 4th rejected (same instant).
    limiter = RateLimiter(rate_per_min=60, burst=3)
    assert [limiter.allow("k", now=0.0) for _ in range(3)] == [True, True, True]
    assert limiter.allow("k", now=0.0) is False


def test_keys_are_isolated():
    limiter = RateLimiter(rate_per_min=60, burst=1)
    assert limiter.allow("alice", now=0.0) is True
    # Bob has his own bucket, unaffected by alice exhausting hers.
    assert limiter.allow("bob", now=0.0) is True
    assert limiter.allow("alice", now=0.0) is False


def test_bucket_refills_with_time():
    # 60/min == 1 token/sec. Empty the bucket, then advance 1s -> 1 token back.
    limiter = RateLimiter(rate_per_min=60, burst=1)
    assert limiter.allow("k", now=0.0) is True
    assert limiter.allow("k", now=0.0) is False
    assert limiter.allow("k", now=1.0) is True
