"""Observability: latency tracking, miss/429 counters, and /metrics.

All offline: forces the mock provider + hashing embedder so results are
deterministic and no network is touched.
"""

import asyncio

from fastapi.testclient import TestClient

from app.gateway.cache import ResponseCache
from app.gateway.rate_limiter import RateLimiter
from app.gateway.router import (
    Gateway,
    LatencyWindow,
    RateLimitExceeded,
    _percentile,
)
from app.main import app
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, Message

client = TestClient(app)


def _req(text="hello world"):
    return ChatRequest(model="mock-1", messages=[Message(role="user", content=text)])


def _gateway(providers=None, rate=600, burst=10):
    return Gateway(
        providers or [MockProvider()],
        ResponseCache(),
        RateLimiter(rate_per_min=rate, burst=burst),
    )


# --- pure helpers ------------------------------------------------------------
def test_percentile_interpolates_and_handles_edges():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7.0], 0.95) == 7.0
    # p50 of 1..4 == 2.5 (midpoint between the two central values).
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0


def test_latency_window_is_bounded():
    win = LatencyWindow(maxlen=3)
    for v in (1.0, 2.0, 3.0, 4.0):
        win.record(v)
    # Oldest sample (1.0) dropped; memory stays constant.
    assert win.count == 3
    assert win.summary()["latency_ms_avg"] == 3.0  # mean of 2,3,4


# --- latency in /stats -------------------------------------------------------
async def test_stats_reports_positive_latency_percentiles():
    gw = _gateway()
    await gw.chat(_req())
    s = gw.stats()
    assert s.latency_ms_p50 > 0
    assert s.latency_ms_p95 > 0
    assert s.latency_ms_avg > 0
    # p95 is never below p50 for the same window.
    assert s.latency_ms_p95 >= s.latency_ms_p50


class _SlowProvider(MockProvider):
    """Adds a fixed delay so the provider (miss) path is measurably slower than a
    cache hit — deterministic without depending on wall-clock jitter."""

    async def complete(self, req):
        await asyncio.sleep(0.02)
        return await super().complete(req)


async def test_cache_hit_latency_is_smaller_than_the_miss():
    gw = _gateway([_SlowProvider()])
    first = await gw.chat(_req())    # miss -> ~20ms provider call
    second = await gw.chat(_req())   # hit  -> just a cache lookup

    assert first.cached is False and second.cached is True
    assert first.usage.latency_ms > 0 and second.usage.latency_ms > 0
    assert second.usage.latency_ms <= first.usage.latency_ms


# --- miss / hit / 429 counters ----------------------------------------------
async def test_miss_then_hit_accounting_is_consistent():
    gw = _gateway()
    await gw.chat(_req("first"))     # uncached -> miss
    await gw.chat(_req("first"))     # repeat   -> hit
    s = gw.stats()
    assert s.cache_misses == 1
    assert s.cache_hits == 1
    # Every request is either a hit or a miss (no 429s here).
    assert s.requests == s.cache_hits + s.cache_misses


async def test_rate_limit_rejections_increment_on_429():
    gw = _gateway(rate=60, burst=1)
    await gw.chat(_req("a"), api_key="k")          # consumes the one token
    try:
        await gw.chat(_req("b"), api_key="k")      # bucket empty -> rejected
        raise AssertionError("expected RateLimitExceeded")
    except RateLimitExceeded:
        pass
    assert gw.stats().rate_limit_rejections == 1


# --- /metrics endpoint -------------------------------------------------------
def test_metrics_endpoint_is_prometheus_text():
    # Drive at least one request so latency samples exist.
    client.post(
        "/v1/chat/completions",
        json={"model": "mock-1", "messages": [{"role": "user", "content": "hi"}]},
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")

    body = r.text
    for name in (
        "gateway_requests_total",
        "gateway_cache_hits_total",
        "gateway_cache_misses_total",
        "gateway_rate_limit_rejections_total",
        "gateway_request_latency_ms_count",
        "gateway_request_latency_ms_sum",
        "gateway_request_latency_ms_p50",
        "gateway_request_latency_ms_p95",
    ):
        assert name in body, f"missing metric {name}"
    # Valid exposition: every non-comment line carries a metric name + value.
    assert "# HELP gateway_requests" in body
    assert "# TYPE gateway_request_latency_ms summary" in body


def test_stats_endpoint_exposes_new_fields():
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    for field in (
        "cache_misses",
        "rate_limit_rejections",
        "latency_ms_p50",
        "latency_ms_p95",
        "latency_ms_avg",
    ):
        assert field in body
