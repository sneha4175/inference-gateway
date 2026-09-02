# Benchmarks

## `cache_latency.py` — cache-hit latency

Measures how much the response cache saves by comparing the gateway's reported
`usage.latency_ms` across three request mixes. It is the measurement foundation
for reproducing the "Don't Break the Cache" prompt-caching work: first prove you
can *measure* the cache effect deterministically offline, then swap in a real
provider to see the cost/TTFT behaviour on large prompts.

### What it measures

| Scenario | Requests | Cache behaviour |
|---|---|---|
| **cold**  | all unique prompts    | 0% hits — every call goes to the provider |
| **warm**  | one prompt repeated   | ~all hits after the first — cache serves the rest |
| **mixed** | each prompt sent twice | ~50% hits — the realistic in-between |

For each it prints p50 / p95 / avg latency, the hit rate, and the cumulative
token counts and cost from `GET /stats`.

### How it works

- Runs **in-process** via FastAPI's `TestClient` — no separate server, fully
  deterministic.
- The shipped `MockProvider` answers instantly, so both a provider call and a
  cache hit would round to ~0 ms and the win would be invisible. The benchmark
  therefore wraps the mock in a **bench-only `SlowProvider`** that sleeps a
  configurable amount (default 50 ms) to stand in for real provider latency. The
  shipped `MockProvider` is left untouched. A cache hit skips the provider, so it
  skips that delay — which is exactly what the numbers expose.
- Each scenario is run against a freshly-built gateway (empty cache, zeroed
  `/stats`) with rate limiting set effectively unlimited, so 200 back-to-back
  requests are never throttled.

### Run it (offline, no API key, no cost)

```bash
python -m benchmarks.cache_latency
```

Knobs (env vars): `BENCH_N` (requests per scenario, default 200),
`BENCH_DELAY_MS` (injected provider delay, default 50).

Example output (offline, 50 ms injected delay, on a laptop):

```
scenario  reqs   hit %    p50 ms    p95 ms    avg ms  prompt tok  compl tok    cost $
-------------------------------------------------------------------------------------
cold       200    0.0%    51.285    51.440    51.238         800       2600    0.0000
warm       200   99.5%     0.012     0.035     0.271           4         13    0.0000
mixed      200   50.0%    25.213    51.377    25.639         300       1200    0.0000
```

Cache hits come back **~4000x faster** than misses here because a hit pays none
of the injected provider delay. Note also the paper's paradox in the `cold` row:
when every prompt is unique the cache never hits, so it adds only lookup/store
overhead and buys nothing — naive "cache everything" is not automatically a win.

### What this does and does NOT show

- **Shows:** the cache-hit speedup and a deterministic, reproducible way to
  *measure* it, plus how the win scales with hit rate.
- **Does NOT show:** the paper's actual cost / time-to-first-token result on
  large prompts. That depends on a **real** provider's prompt-caching behaviour
  and cannot be reproduced against the offline mock (whose delay is a fixed
  constant, not a function of prompt size).

## Running against a real provider (needs a small API budget)

The gateway already targets the OpenAI-compatible wire format, so you can point
it at OpenAI, Together, Groq, OpenRouter, a local vLLM, etc. To reproduce the
paper's cost/TTFT result on large prompts, run the gateway against a real
endpoint and drive traffic through it:

```bash
export PROVIDER=openai
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
# use a real model id (e.g. gpt-4o-mini) instead of mock-1 in the request body
```

> ⚠️ **This spends real money.** Each miss is a billed provider call. Start with a
> small `BENCH_N` and a cheap model, and watch `total_cost_usd` in `/stats`.

To exercise the paper's scenario specifically, send **large shared-prefix
prompts** (a long system prompt / retrieved context reused across many requests)
so the provider's prompt cache can engage, and compare cost and latency for the
cold vs. warm mixes. With a real provider the `cost $` column becomes non-zero
and the warm mix should show both lower latency and lower cost per request — the
result the offline mock cannot demonstrate.
