"""Per-key rate limiting via the token-bucket algorithm.

Why rate-limit an LLM gateway specifically: every downstream call costs real
money and upstream providers enforce their own quotas. Without a limiter, one
buggy client (or an abusive key) can burn the budget or get everyone throttled.

Token bucket (vs a fixed window): each key owns a bucket that refills at a
steady rate up to a burst capacity. Every request removes one token; if the
bucket is empty the request is rejected. It allows short bursts while capping the
long-run average — smoother and fairer than resetting a counter every 60s.
"""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self, rate_per_min: int, burst: int | None = None) -> None:
        # Refill speed in tokens/second.
        self.refill_per_sec = rate_per_min / 60.0
        # Bucket size = max burst. Defaults to a full minute's worth.
        self.capacity = float(burst if burst is not None else rate_per_min)
        # key -> (tokens_remaining, last_refill_timestamp)
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        """Consume one token for `key`. Return True if allowed, False if limited.

        `now` is injectable so tests can advance time deterministically instead
        of sleeping.
        """
        now = time.monotonic() if now is None else now
        tokens, last = self._buckets.get(key, (self.capacity, now))

        # Refill based on elapsed time, capped at capacity.
        tokens = min(self.capacity, tokens + (now - last) * self.refill_per_sec)

        if tokens >= 1.0:
            self._buckets[key] = (tokens - 1.0, now)
            return True

        # Rejected: remember the (partial) refill so the clock keeps ticking.
        self._buckets[key] = (tokens, now)
        return False
