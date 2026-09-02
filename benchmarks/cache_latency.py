"""Cache-latency benchmark for the inference gateway.

WHAT THIS MEASURES
------------------
Response caching only pays off when the same request is seen more than once. This
harness measures, in-process (FastAPI ``TestClient``, no separate server), the
per-request ``usage.latency_ms`` the gateway reports across three request mixes:

  1. cold  — every request unique              -> 0% cache hits (all misses)
  2. warm  — one request repeated N times       -> ~all hits after the first
  3. mixed — every prompt sent twice, shuffled  -> ~50% hits

and prints p50 / p95 / avg latency, the hit rate, and the token "cost" pulled
from ``GET /stats`` for each.

WHY AN INJECTED DELAY
---------------------
The shipped ``MockProvider`` runs fully offline and answers instantly, so a real
provider call and a cache hit would both round to ~0 ms and the cache win would be
invisible. To make the effect measurable we wrap the mock in a *bench-only*
``SlowProvider`` that sleeps a configurable amount (default 50 ms) to stand in for
real provider latency. This does NOT touch the shipped ``MockProvider`` — it is a
separate provider used only here. A cache hit skips the provider entirely, so it
skips the sleep, which is exactly the latency the benchmark exposes.

WHAT IT DOES *NOT* SHOW
-----------------------
The offline run demonstrates the measurement methodology and the raw cache-hit
speedup on a fixed per-call delay. It does NOT reproduce the "Don't Break the
Cache" paper's cost/TTFT result, which depends on a *real* provider's
prompt-caching behaviour on large prompts. See ``benchmarks/README.md`` for how to
point this at a real OpenAI-compatible endpoint.

RUN
---
    python -m benchmarks.cache_latency

Env knobs: ``BENCH_N`` (requests per scenario, default 200),
``BENCH_DELAY_MS`` (injected provider delay, default 50).
"""

from __future__ import annotations

import math
import os

from fastapi.testclient import TestClient

from app.gateway.cache import ResponseCache
from app.gateway.rate_limiter import RateLimiter
from app.gateway.router import Gateway
from app.main import app, get_gateway
from app.providers.mock import MockProvider
from app.schemas import ChatRequest, ChatResponse

# Bench defaults. Kept small enough to run in a few seconds, large enough for
# stable percentiles.
N = int(os.environ.get("BENCH_N", "200"))
DELAY_MS = float(os.environ.get("BENCH_DELAY_MS", "50"))
MODEL = "mock-1"


class SlowProvider(MockProvider):
    """Bench-only provider: a MockProvider that sleeps before answering.

    Stands in for real network/provider latency so the value of skipping the
    provider (a cache hit) is measurable. Not used anywhere in the shipped app.
    """

    def __init__(self, delay_ms: float, name: str = "slow-mock") -> None:
        super().__init__(name=name)
        self._delay_s = delay_ms / 1000.0

    async def complete(self, req: ChatRequest) -> ChatResponse:
        import asyncio

        await asyncio.sleep(self._delay_s)
        return await super().complete(req)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolated percentile — same method the gateway's /stats uses."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 3)
    k = (len(sorted_vals) - 1) * pct
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return round(sorted_vals[int(k)], 3)
    return round(sorted_vals[lo] * (hi - k) + sorted_vals[hi] * (k - lo), 3)


def _build_client() -> TestClient:
    """A TestClient whose gateway uses the slow provider and no rate limiting.

    We override the app's ``get_gateway`` dependency with a freshly-built gateway
    so each scenario starts from an empty cache and zeroed /stats counters. The
    rate limiter is set effectively unlimited so 200 back-to-back requests are
    never throttled (that is a different concern from caching).
    """
    gateway = Gateway(
        providers=[SlowProvider(DELAY_MS)],
        cache=ResponseCache(ttl_seconds=3600, max_size=10 * N),
        limiter=RateLimiter(rate_per_min=10_000_000, burst=10_000_000),
    )
    app.dependency_overrides[get_gateway] = lambda: gateway
    return TestClient(app)


def _chat(client: TestClient, content: str) -> float:
    """Send one chat request; return the gateway-reported latency in ms."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": content}]},
    )
    resp.raise_for_status()
    return resp.json()["usage"]["latency_ms"]


def _prompts_cold(n: int) -> list[str]:
    """Every prompt unique -> guaranteed 0% hit rate."""
    return [f"unique request number {i}" for i in range(n)]


def _prompts_warm(n: int) -> list[str]:
    """One prompt repeated -> first is a miss, the rest are hits."""
    return ["the same repeated request"] * n


def _prompts_mixed(n: int) -> list[str]:
    """Each of n/2 prompts sent exactly twice, deterministically interleaved.

    First occurrence of each is a miss, the second a hit -> ~50% hit rate,
    independent of order.
    """
    half = n // 2
    base = [f"mixed request {i}" for i in range(half)]
    # Interleave [p0, p0, p1, p1, ...] so hits and misses are spread through the
    # run rather than clustered — a more realistic arrival pattern than "all
    # misses then all hits".
    out: list[str] = []
    for p in base:
        out.append(p)
        out.append(p)
    return out[:n]


def run_scenario(name: str, prompts: list[str]) -> dict[str, float]:
    client = _build_client()
    latencies = [_chat(client, p) for p in prompts]
    stats = client.get("/stats").json()
    app.dependency_overrides.clear()

    latencies.sort()
    hits = stats["cache_hits"]
    total = stats["cache_hits"] + stats["cache_misses"]
    return {
        "name": name,
        "requests": total,
        "hit_rate": (hits / total * 100.0) if total else 0.0,
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "avg": round(sum(latencies) / len(latencies), 3),
        "prompt_tokens": stats["total_prompt_tokens"],
        "completion_tokens": stats["total_completion_tokens"],
        "cost_usd": stats["total_cost_usd"],
    }


def _print_table(rows: list[dict[str, float]]) -> None:
    header = (
        f"{'scenario':<8} {'reqs':>5} {'hit %':>7} "
        f"{'p50 ms':>9} {'p95 ms':>9} {'avg ms':>9} "
        f"{'prompt tok':>11} {'compl tok':>10} {'cost $':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<8} {r['requests']:>5} {r['hit_rate']:>6.1f}% "
            f"{r['p50']:>9.3f} {r['p95']:>9.3f} {r['avg']:>9.3f} "
            f"{r['prompt_tokens']:>11} {r['completion_tokens']:>10} "
            f"{r['cost_usd']:>9.4f}"
        )


def main() -> None:
    print(
        f"cache-latency benchmark | N={N} requests/scenario | "
        f"injected provider delay={DELAY_MS:.0f} ms | provider=slow-mock (offline)\n"
    )
    rows = [
        run_scenario("cold", _prompts_cold(N)),
        run_scenario("warm", _prompts_warm(N)),
        run_scenario("mixed", _prompts_mixed(N)),
    ]
    _print_table(rows)

    cold = next(r for r in rows if r["name"] == "cold")
    warm = next(r for r in rows if r["name"] == "warm")
    speedup = (cold["p50"] / warm["p50"]) if warm["p50"] else float("inf")

    print(
        "\nReading the table:\n"
        f"  * warm (cache hit) p50 is {speedup:,.0f}x faster than cold (cache "
        "miss) p50 — the cache serves repeats without paying the provider delay.\n"
        "  * cold is 0% hits: every prompt is unique, so the cache never helps and\n"
        "    only adds a lookup + store on each call — the paper's paradox setup\n"
        "    (naive caching of all-unique prompts is pure overhead, no win).\n"
        "  * mixed (~50% hits) lands between the two: the win scales with hit rate.\n"
        "\nNote: this shows the cache-hit speedup and the measurement method on a\n"
        "fixed injected delay. It does NOT reproduce the paper's cost/TTFT result on\n"
        "large prompts — that needs a real provider (see benchmarks/README.md)."
    )


if __name__ == "__main__":
    main()
