"""The gateway orchestrator: ties the pieces together for one request.

Pipeline for a chat call:

    rate limit  ->  cache lookup  ->  provider (with fallback)  ->  cache store

Order matters and is a deliberate choice:
  * rate limit FIRST so an abusive key is rejected as cheaply as possible.
  * cache SECOND so repeated prompts skip the provider entirely.
  * provider fallback LAST: try providers in order, and on ProviderError move to
    the next one. This is what keeps the gateway up when a single vendor has an
    outage.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import AsyncIterator

from app.gateway.cache import ResponseCache, cache_key
from app.gateway.rate_limiter import RateLimiter
from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, ChatResponse, Stats


class RateLimitExceeded(Exception):
    """Raised when a key is over its rate limit -> HTTP 429."""


class AllProvidersFailed(Exception):
    """Raised when every provider in the fallback chain errored -> HTTP 502."""


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list. Empty -> 0.0."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    k = (len(sorted_vals) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return round(sorted_vals[int(k)], 3)
    val = sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo)
    return round(val, 3)


class LatencyWindow:
    """Bounded rolling window of recent per-request latencies (milliseconds).

    A deque with maxlen keeps memory constant: once full, appending drops the
    oldest sample. Percentiles are computed on demand from the live window, so
    /stats and /metrics always agree (single source of truth).
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)

    def record(self, ms: float) -> None:
        self._samples.append(ms)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def total(self) -> float:
        return sum(self._samples)

    def summary(self) -> dict[str, float]:
        vals = sorted(self._samples)
        avg = round(sum(vals) / len(vals), 3) if vals else 0.0
        return {
            "latency_ms_p50": _percentile(vals, 0.50),
            "latency_ms_p95": _percentile(vals, 0.95),
            "latency_ms_avg": avg,
        }


class Gateway:
    def __init__(
        self,
        providers: list[Provider],
        cache: ResponseCache,
        limiter: RateLimiter,
    ) -> None:
        if not providers:
            raise ValueError("Gateway needs at least one provider")
        # Ordered list == fallback chain. providers[0] is primary.
        self.providers = providers
        self.cache = cache
        self.limiter = limiter
        # Running totals surfaced on /stats.
        self._stats = {
            "requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "provider_errors": 0,
            "rate_limit_rejections": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        }
        # Rolling window of recent latencies feeding the /stats percentiles.
        self.latencies = LatencyWindow()

    def _record(self, resp: ChatResponse) -> None:
        self._stats["total_prompt_tokens"] += resp.usage.prompt_tokens
        self._stats["total_completion_tokens"] += resp.usage.completion_tokens
        self._stats["total_cost_usd"] = round(
            self._stats["total_cost_usd"] + resp.usage.cost_usd, 6
        )

    def _finish(self, resp: ChatResponse, start: float) -> ChatResponse:
        """Stamp the served response with its wall-clock latency and record it.

        Called on both the cache-hit and provider paths so a cache hit — which
        does far less work — genuinely reports a smaller latency than a miss.
        """
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        resp.usage.latency_ms = round(elapsed_ms, 3)
        self.latencies.record(elapsed_ms)
        return resp

    async def chat(self, req: ChatRequest, api_key: str = "anonymous") -> ChatResponse:
        start = time.perf_counter()
        self._stats["requests"] += 1

        # 1) Rate limit.
        if not self.limiter.allow(api_key):
            self._stats["rate_limit_rejections"] += 1
            raise RateLimitExceeded(f"rate limit exceeded for key '{api_key}'")

        # 2) Cache.
        key = cache_key(req)
        hit = self.cache.get(key)
        if hit is not None:
            self._stats["cache_hits"] += 1
            # deep copy so stamping latency_ms never mutates the cached entry.
            resp = hit.model_copy(update={"cached": True}, deep=True)
            return self._finish(resp, start)

        self._stats["cache_misses"] += 1

        # 3) Provider call with fallback.
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                resp = await provider.complete(req)
            except ProviderError as exc:
                self._stats["provider_errors"] += 1
                last_error = exc
                continue  # try the next provider in the chain
            self.cache.set(key, resp)
            self._record(resp)
            return self._finish(resp, start)

        raise AllProvidersFailed(f"all providers failed; last error: {last_error}")

    async def stream(
        self, req: ChatRequest, api_key: str = "anonymous"
    ) -> AsyncIterator[str]:
        """Streaming path. Not cached (deltas arrive incrementally) but it still
        rate-limits and falls back on the *initial* connection error."""
        self._stats["requests"] += 1
        if not self.limiter.allow(api_key):
            self._stats["rate_limit_rejections"] += 1
            raise RateLimitExceeded(f"rate limit exceeded for key '{api_key}'")

        last_error: Exception | None = None
        for provider in self.providers:
            try:
                agen = provider.stream(req)
                # Pull the first delta eagerly so a connection failure surfaces
                # here and we can still fall back before streaming to the client.
                first = await agen.__anext__()
            except ProviderError as exc:
                self._stats["provider_errors"] += 1
                last_error = exc
                continue
            except StopAsyncIteration:
                return

            async def _gen(first_delta: str, gen):
                yield first_delta
                async for delta in gen:
                    yield delta

            async for delta in _gen(first, agen):
                yield delta
            return

        raise AllProvidersFailed(f"all providers failed; last error: {last_error}")

    def stats(self) -> Stats:
        return Stats(**self._stats, **self.latencies.summary())
