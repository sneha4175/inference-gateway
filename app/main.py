"""FastAPI application = the HTTP surface of the gateway.

Routes:
  GET  /health                 liveness
  GET  /stats                  request/cache/token/cost/latency counters (JSON)
  GET  /metrics                Prometheus text exposition of the same counters
  POST /v1/chat/completions    proxy a chat request (streaming optional)
  POST /rag/ingest             index documents
  POST /rag/query              retrieve + generate

The gateway and RAG pipeline are built once at startup from env config and shared
across requests (they hold the cache, limiter buckets and vector store).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.config import Settings, build_gateway, build_rag
from app.gateway.metrics import build_registry, render
from app.gateway.router import AllProvidersFailed, Gateway, RateLimitExceeded
from app.rag.pipeline import RagPipeline
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    Stats,
)

app = FastAPI(
    title="AI Inference Gateway",
    description="LLM proxy with rate limiting, caching, fallback + a RAG route.",
    version="0.2.0",
)

# Composition root: build shared singletons at import/startup.
_settings = Settings.from_env()
_gateway = build_gateway(_settings)
_rag = build_rag(_settings, _gateway)
# Prometheus registry bound to the shared gateway (scraped by GET /metrics).
_metrics_registry = build_registry(_gateway)


# Dependency-injection seams. Tests override these to inject their own stack.
def get_gateway() -> Gateway:
    return _gateway


def get_rag() -> RagPipeline:
    return _rag


def api_key(x_api_key: str | None = Header(default=None)) -> str:
    """Identify the caller for rate limiting. Anonymous if no header given.

    Real deployments would authenticate the key; here it only scopes the rate
    limiter's per-key bucket.
    """
    return x_api_key or "anonymous"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/stats", response_model=Stats)
def stats(gateway: Gateway = Depends(get_gateway)) -> Stats:
    return gateway.stats()


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint. Same counters as /stats, machine-readable."""
    body, content_type = render(_metrics_registry)
    return Response(content=body, media_type=content_type)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    req: ChatRequest,
    gateway: Gateway = Depends(get_gateway),
    key: str = Depends(api_key),
):
    if req.stream:
        # Stream plain text deltas. Kept simple (not SSE-framed) so the offline
        # demo is easy to read; production would emit OpenAI-style SSE.
        async def gen():
            try:
                async for delta in gateway.stream(req, api_key=key):
                    yield delta
            except RateLimitExceeded as exc:
                raise HTTPException(status_code=429, detail=str(exc))
            except AllProvidersFailed as exc:
                raise HTTPException(status_code=502, detail=str(exc))

        return StreamingResponse(gen(), media_type="text/plain")

    try:
        return await gateway.chat(req, api_key=key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except AllProvidersFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/rag/ingest", response_model=IngestResponse)
def rag_ingest(req: IngestRequest, rag: RagPipeline = Depends(get_rag)) -> IngestResponse:
    try:
        return rag.ingest(req.documents, req.doc_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/rag/query", response_model=RagQueryResponse)
async def rag_query(
    req: RagQueryRequest,
    rag: RagPipeline = Depends(get_rag),
    key: str = Depends(api_key),
):
    try:
        return await rag.query(req.query, top_k=req.top_k, model=req.model, api_key=key)
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except AllProvidersFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc))
