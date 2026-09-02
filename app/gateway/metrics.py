"""Prometheus exposition for the gateway.

Why Prometheus text format: `/stats` is a convenient JSON snapshot for humans,
but the de-facto machine-readable contract for metrics is the Prometheus text
exposition format, which any scraper (Prometheus, Grafana Agent, OTEL collector)
can ingest without bespoke glue. We use the standard `prometheus_client` library
(one dependency) rather than hand-rolling the format — it gets the escaping,
`# HELP`/`# TYPE` headers and content-type exactly right.

Design: a *custom collector* reads the live Gateway on each scrape instead of a
second set of counters updated alongside `_stats`. That keeps the Gateway the
single source of truth, so `/metrics` and `/stats` can never drift apart.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    SummaryMetricFamily,
)

from app.gateway.router import Gateway


class GatewayCollector:
    """Translates a Gateway's live counters + latency window into Prometheus
    metric families at scrape time."""

    def __init__(self, gateway: Gateway) -> None:
        self.gateway = gateway

    def collect(self):
        s = self.gateway._stats

        counters = [
            ("gateway_requests", "Total chat/RAG requests received.", s["requests"]),
            ("gateway_cache_hits", "Requests served from the response cache.",
             s["cache_hits"]),
            ("gateway_cache_misses", "Requests that missed the cache and hit a "
             "provider.", s["cache_misses"]),
            ("gateway_provider_errors", "Provider call failures (drives fallback).",
             s["provider_errors"]),
            ("gateway_rate_limit_rejections", "Requests rejected with HTTP 429 by "
             "the token-bucket limiter.", s["rate_limit_rejections"]),
            ("gateway_prompt_tokens", "Cumulative prompt tokens processed.",
             s["total_prompt_tokens"]),
            ("gateway_completion_tokens", "Cumulative completion tokens generated.",
             s["total_completion_tokens"]),
        ]
        for name, doc, value in counters:
            yield CounterMetricFamily(name, doc, value=value)

        # Cost is a float running total; still a monotonic counter.
        yield CounterMetricFamily(
            "gateway_cost_usd", "Cumulative estimated spend in USD.",
            value=s["total_cost_usd"],
        )

        # Latency as a summary (_sum/_count) plus explicit quantile gauges, all
        # derived from the same rolling window that /stats reports.
        window = self.gateway.latencies
        lat = window.summary()
        yield SummaryMetricFamily(
            "gateway_request_latency_ms",
            "Per-request latency in milliseconds over a rolling window.",
            count_value=window.count,
            sum_value=window.total,
        )
        yield GaugeMetricFamily(
            "gateway_request_latency_ms_p50",
            "Median request latency (ms) over the rolling window.",
            value=lat["latency_ms_p50"],
        )
        yield GaugeMetricFamily(
            "gateway_request_latency_ms_p95",
            "95th-percentile request latency (ms) over the rolling window.",
            value=lat["latency_ms_p95"],
        )
        yield GaugeMetricFamily(
            "gateway_request_latency_ms_avg",
            "Mean request latency (ms) over the rolling window.",
            value=lat["latency_ms_avg"],
        )


def build_registry(gateway: Gateway) -> CollectorRegistry:
    """A private registry holding only this gateway's collector.

    Using a fresh registry (instead of the global default) means importing this
    module has no global side effects and multiple gateways/tests never collide.
    """
    registry = CollectorRegistry()
    registry.register(GatewayCollector(gateway))
    return registry


def render(registry: CollectorRegistry) -> tuple[bytes, str]:
    """Return (body, content_type) for a Prometheus scrape response."""
    return generate_latest(registry), CONTENT_TYPE_LATEST
