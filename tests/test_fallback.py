"""Provider fallback + rate-limit rejection + token/cost accounting."""

import pytest

from app.gateway.cache import ResponseCache
from app.gateway.rate_limiter import RateLimiter
from app.gateway.router import AllProvidersFailed, Gateway, RateLimitExceeded
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, Message


def _req(text="hello world"):
    return ChatRequest(model="mock-1", messages=[Message(role="user", content=text)])


def _gateway(providers, rate=600, burst=10):
    return Gateway(providers, ResponseCache(), RateLimiter(rate_per_min=rate, burst=burst))


async def test_falls_back_to_healthy_provider():
    dead = MockProvider(name="primary", should_fail=True)
    alive = MockProvider(name="secondary")
    gateway = _gateway([dead, alive])

    resp = await gateway.chat(_req())

    # Primary errored, secondary answered.
    assert resp.provider == "secondary"
    assert gateway.stats().provider_errors == 1


async def test_all_providers_failing_raises():
    gateway = _gateway([MockProvider(name="a", should_fail=True),
                        MockProvider(name="b", should_fail=True)])
    with pytest.raises(AllProvidersFailed):
        await gateway.chat(_req())


async def test_rate_limit_rejects_when_exhausted():
    gateway = _gateway([MockProvider()], rate=60, burst=1)
    await gateway.chat(_req("a"), api_key="key1")  # consumes the one token
    with pytest.raises(RateLimitExceeded):
        await gateway.chat(_req("b"), api_key="key1")


async def test_usage_counts_tokens_and_is_free_for_mock():
    gateway = _gateway([MockProvider()])
    resp = await gateway.chat(_req("one two three"))
    assert resp.usage.prompt_tokens == 3          # "one", "two", "three"
    assert resp.usage.completion_tokens > 0
    assert resp.usage.total_tokens == (
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    )
    assert resp.usage.cost_usd == 0.0             # mock-1 is priced at $0
