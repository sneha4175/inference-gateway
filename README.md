# Inference Gateway

A small but real **AI inference gateway** written from scratch in Python / FastAPI.
It sits in front of LLM providers and adds the operational glue you need before
LLM calls are safe to expose to real traffic: **provider abstraction, per-key
rate limiting, response caching, token/cost/latency tracking, streaming, provider
fallback, and observability (`/stats` JSON + a Prometheus `/metrics` endpoint).**
It also ships a **RAG route** (`/rag`) — document ingest → chunk →
embed → vector store → retrieve → augment → generate — exposed through the *same*
gateway so RAG answers inherit all of the above.

It is a portfolio project, and it is honest about that: it runs and is fully
tested **offline** with a deterministic mock provider and a hashing embedder, and
you can swap in a real provider/embedder with two environment variables. It is
not an enterprise product clone — see [Trade-offs](#trade-offs--what-production-would-add).

---

## Why an LLM gateway?

Calling a provider SDK directly from every service works in a demo and breaks in
production. Every call costs money **per token**, providers rate-limit and
occasionally go down, identical prompts get re-sent, and nobody can see what is
being spent. A gateway centralises those concerns in one place — the same reason
teams put an API gateway in front of microservices, applied to LLM traffic.

---

## Architecture

```
                                     ┌──────────────────────────────────────────────┐
                                     │                 FastAPI app                  │
                                     │                                              │
   client ──► /v1/chat/completions ──┼─► Gateway pipeline                           │
                                     │     1. rate limit (token bucket, per key)    │
                                     │     2. cache lookup (hash of model+messages) │
                                     │     3. provider(s): one, or a fallback chain │
                                     │     4. cache store + cost/token/latency stats│
                                     │                                              │
   client ──► /rag/query ────────────┼─► RAG pipeline                               │
                                     │     embed query               Provider(s):   │
                                     │     retrieve top-k <- VectorStore  |- Mock   │
                                     │     augment prompt (NumPy cosine)  \- OpenAI-│
                                     │     generate ------------------->    compat  │
                                     │                                              │
   client ──► /rag/ingest ───────────┼─► chunk -> embed -> VectorStore              │
                                     └──────────────────────────────────────────────┘
```

The offline default runs a **single** mock provider — step 3 is a one-element
chain. A real fallback chain (primary provider + a mock safety net) exists only
with `PROVIDER=openai`, and is exercised directly in `tests/test_fallback.py`.

Two halves, one pipeline:

* **Gateway half** — `app/gateway/` (rate limiter, cache, cost, router) +
  `app/providers/` (the pluggable backends behind one interface).
* **RAG half** — `app/rag/` (chunk, embed, vector store, pipeline). The RAG
  pipeline calls the **same** `Gateway.chat()` for generation, so retrieval
  answers are rate-limited, cached, costed and fault-tolerant for free.

### Request flow (chat)

`rate limit → cache → provider (with fallback) → cache store`. The order is
deliberate: reject abusive keys most cheaply first, serve repeats from cache
second, and only then spend a provider call — trying each provider in order and
moving to the next on error.

### Request flow (RAG)

`ingest`: split each document into overlapping token chunks, embed each chunk,
store the vectors. `query`: embed the question, cosine-search the top-k chunks,
stuff them into the prompt as context, and generate through the gateway.

---

## Running it

### Offline (default — no API key, no cost, no downloads)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Everything defaults to the **mock provider** + **hashing embedder**, so it works
with zero configuration.

```bash
# chat
curl -s localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"mock-1","messages":[{"role":"user","content":"hello"}]}'

# RAG: ingest then ask
curl -s localhost:8000/rag/ingest -H 'content-type: application/json' \
  -d '{"documents":["The capital of Australia is Canberra."],"doc_ids":["geo"]}'

curl -s localhost:8000/rag/query -H 'content-type: application/json' \
  -d '{"query":"What is the capital of Australia?","top_k":1}'

curl -s localhost:8000/stats     # request / cache / token / cost / latency (JSON)
curl -s localhost:8000/metrics   # the same counters in Prometheus text format
```

The mock provider **echoes the prompt it is given**. That is intentional: because
the RAG pipeline injects the retrieved context into that prompt, the retrieved
fact ("Canberra") shows up in the answer — which is exactly what lets the offline
tests *prove* retrieval worked without a real model.

### With a real provider / embedder

The real backends are pluggable and target the **OpenAI-compatible** wire format,
so the same code works with OpenAI, Together, Groq, OpenRouter, a local vLLM, etc.
via `OPENAI_BASE_URL`.

```bash
export PROVIDER=openai
export EMBEDDER=openai            # optional; leave as hashing to stay offline for RAG
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # or any compatible endpoint
uvicorn app.main:app
# now use model:"gpt-4o-mini" instead of "mock-1"
```

| Env var | Default | Meaning |
|---|---|---|
| `PROVIDER` | `mock` | `mock` or `openai` |
| `EMBEDDER` | `hashing` | `hashing` (offline) or `openai` |
| `OPENAI_API_KEY` | – | required when either is `openai` |
| `OPENAI_BASE_URL` | OpenAI | any OpenAI-compatible endpoint |
| `RATE_LIMIT_PER_MIN` | `60` | tokens refilled per key per minute |
| `RATE_LIMIT_BURST` | `10` | bucket capacity (max burst) |
| `CACHE_TTL_SECONDS` | `300` | response cache TTL |

### Docker

```bash
docker build -t inference-gateway .
docker run -p 8000:8000 inference-gateway      # offline mock stack by default
```

### Tests

```bash
pytest        # 29 tests, all offline
```

Covers: rate-limiter burst + refill, cache hit/miss + TTL, fallback on provider
error, "all providers failed", token counting + cost, RAG retrieval returns the
right chunk, an end-to-end `/rag` call through the FastAPI app, and the
observability additions (latency percentiles, miss/429 counters, `/metrics`).

---

## Metrics & Observability

Two views of the same live counters, both offline and dependency-light:

* **`GET /stats`** — a JSON snapshot for humans and quick `curl` checks.
* **`GET /metrics`** — Prometheus text exposition (via `prometheus-client`) for
  scrapers. A custom collector reads the gateway on each scrape, so `/stats` and
  `/metrics` are guaranteed never to drift apart.

`/stats` fields:

| Field | Meaning |
|---|---|
| `requests` | total chat/RAG requests received |
| `cache_hits` / `cache_misses` | responses served from cache vs. forwarded to a provider |
| `provider_errors` | provider call failures (each drives a fallback attempt) |
| `rate_limit_rejections` | requests rejected with HTTP 429 by the token bucket |
| `total_prompt_tokens` / `total_completion_tokens` | cumulative token counts |
| `total_cost_usd` | cumulative estimated spend ($0 on the mock model) |
| `latency_ms_p50` / `latency_ms_p95` / `latency_ms_avg` | request latency over a bounded rolling window (last 1000 samples) |

Latency is wall-clock (`time.perf_counter`) around the whole gateway pipeline, so
a cache hit — which skips the provider — records a genuinely smaller latency than
the miss that populated it. Each per-request response also carries its own
`usage.latency_ms`. `/metrics` exposes the counters as Prometheus counters plus a
`gateway_request_latency_ms` summary (`_count`/`_sum`) and p50/p95/avg gauges.

---

## Design choices worth calling out

* **One tokenizer** (word/punctuation regex) shared by cost tracking and the
  embedder. It approximates provider BPE billing — good enough to demonstrate the
  mechanics, and it avoids a heavy `tiktoken`-style dependency + model download.
* **Hashing embedder** (the "feature hashing" trick): each token hashes to a
  bucket in a fixed vector. Deterministic, offline, captures *lexical* overlap.
  It does **not** capture semantics (synonyms) — an honest limitation, swapped out
  by `EMBEDDER=openai`.
* **NumPy brute-force vector store** instead of FAISS/pgvector: exact, tiny, no
  heavy deps. Linear scan is fine at portfolio scale; an approximate index only
  pays off at millions of vectors.

## Trade-offs — what production would add

| Area | This project | Production |
|---|---|---|
| Cache / rate-limit state | in-process (per instance) | Redis, shared across replicas |
| Vector store | NumPy in memory, non-persistent | pgvector / Qdrant / FAISS, persisted |
| Tokenizer | regex approximation | real BPE (`tiktoken`) per model |
| Auth | header echoed as identity | real API-key auth, quotas, tenants |
| Caching correctness | caches all responses | opt-in / keyed on temperature (random outputs shouldn't cache) |
| Observability | `/stats` JSON + Prometheus `/metrics` (counters + latency percentiles) | + distributed tracing, structured logs, alerting |
| Prompt-injection defense | a "use only the context" system prompt | input/output filtering, allow-lists, eval harness |
| Retrieval quality | top-k cosine, no eval | reranking, hybrid search, an eval set (recall@k) |

---

## Layout

```
app/
  main.py            FastAPI routes + composition root
  config.py          env config + object wiring (pluggable provider/embedder)
  schemas.py         pydantic request/response models
  util.py            shared tokenizer
  providers/         base interface, mock (offline), openai-compatible (real)
  gateway/           rate_limiter, cache, cost, router (the pipeline), metrics (Prometheus)
  rag/               chunk, embed, store (NumPy cosine), pipeline
tests/               29 offline tests
```

This project deliberately extends the ideas from an earlier Go API gateway
(`gateway-pro`) into AI infrastructure: same gateway concerns (routing, limiting,
caching, fallback), new domain (tokens, cost, retrieval).
