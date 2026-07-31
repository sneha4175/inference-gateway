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

from collections.abc import AsyncIterator

from app.gateway.cache import ResponseCache, cache_key
from app.gateway.rate_limiter import RateLimiter
from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, ChatResponse, Stats


class RateLimitExceeded(Exception):
    """Raised when a key is over its rate limit -> HTTP 429."""


class AllProvidersFailed(Exception):
    """Raised when every provider in the fallback chain errored -> HTTP 502."""


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
            "provider_errors": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost_usd": 0.0,
        }

    def _record(self, resp: ChatResponse) -> None:
        self._stats["total_prompt_tokens"] += resp.usage.prompt_tokens
        self._stats["total_completion_tokens"] += resp.usage.completion_tokens
        self._stats["total_cost_usd"] = round(
            self._stats["total_cost_usd"] + resp.usage.cost_usd, 6
        )

    async def chat(self, req: ChatRequest, api_key: str = "anonymous") -> ChatResponse:
        self._stats["requests"] += 1

        # 1) Rate limit.
        if not self.limiter.allow(api_key):
            raise RateLimitExceeded(f"rate limit exceeded for key '{api_key}'")

        # 2) Cache.
        key = cache_key(req)
        hit = self.cache.get(key)
        if hit is not None:
            self._stats["cache_hits"] += 1
            return hit.model_copy(update={"cached": True})

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
            return resp

        raise AllProvidersFailed(f"all providers failed; last error: {last_error}")

    async def stream(
        self, req: ChatRequest, api_key: str = "anonymous"
    ) -> AsyncIterator[str]:
        """Streaming path. Not cached (deltas arrive incrementally) but it still
        rate-limits and falls back on the *initial* connection error."""
        self._stats["requests"] += 1
        if not self.limiter.allow(api_key):
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
        return Stats(**self._stats)
