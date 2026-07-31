"""Response cache: miss then hit, TTL expiry, and the gateway serves from cache
without calling the provider a second time."""

from app.gateway.cache import ResponseCache, cache_key
from app.gateway.rate_limiter import RateLimiter
from app.gateway.router import Gateway
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, Message


def _req(text="hello"):
    return ChatRequest(model="mock-1", messages=[Message(role="user", content=text)])


def test_same_request_hashes_equal():
    assert cache_key(_req()) == cache_key(_req())
    assert cache_key(_req("a")) != cache_key(_req("b"))


def test_ttl_expiry():
    from app.schemas import ChatResponse, Usage

    cache = ResponseCache(ttl_seconds=10)
    resp = ChatResponse(
        id="1", model="mock-1", provider="mock", content="hi",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0),
    )
    cache.set("k", resp, now=0.0)
    assert cache.get("k", now=5.0) is not None      # still fresh
    assert cache.get("k", now=10.0) is None          # expired at ttl


class CountingProvider(MockProvider):
    """Mock that records how many times complete() actually ran."""

    def __init__(self):
        super().__init__(name="counter")
        self.calls = 0

    async def complete(self, req):
        self.calls += 1
        return await super().complete(req)


async def test_gateway_serves_second_call_from_cache():
    provider = CountingProvider()
    gateway = Gateway([provider], ResponseCache(), RateLimiter(rate_per_min=600, burst=10))

    first = await gateway.chat(_req())
    second = await gateway.chat(_req())

    assert provider.calls == 1        # provider hit only once
    assert first.cached is False
    assert second.cached is True
    assert second.content == first.content
