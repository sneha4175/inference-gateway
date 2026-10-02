"""FastAPI application = the HTTP surface of the gateway.

Routes:
  GET  /health                 liveness
  GET  /stats                  request/cache/token/cost/latency counters (JSON)
  GET  /metrics                Prometheus text exposition of the same counters
  POST /v1/chat/completions    proxy a chat request (streaming optional)
  POST /rag/ingest             index documents
  POST /rag/query              retrieve + generate
  POST /tts                    synthesise text to speech audio
  POST /rag/speak              retrieve + generate, returned as speech

The gateway, RAG pipeline and voice engine are built once at startup from env
config and shared across requests (they hold the cache, limiter buckets, vector
store and TTS latency window).
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.config import Settings, build_gateway, build_rag, build_tts
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
    RagSpeakRequest,
    Stats,
    TTSRequest,
)
from app.tts.base import TTSError
from app.tts.engine import TTSEngine

app = FastAPI(
    title="AI Inference Gateway",
    description="LLM proxy with rate limiting, caching, fallback, a RAG route "
    "and an ElevenLabs text-to-speech layer.",
    version="0.3.0",
)

# Composition root: build shared singletons at import/startup.
_settings = Settings.from_env()
_gateway = build_gateway(_settings)
_rag = build_rag(_settings, _gateway)
_tts = build_tts(_settings)
# Prometheus registry bound to the shared gateway + voice engine (GET /metrics).
_metrics_registry = build_registry(_gateway, _tts)


# Dependency-injection seams. Tests override these to inject their own stack.
def get_gateway() -> Gateway:
    return _gateway


def get_rag() -> RagPipeline:
    return _rag


def get_tts() -> TTSEngine:
    return _tts


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
def stats(
    gateway: Gateway = Depends(get_gateway),
    tts: TTSEngine = Depends(get_tts),
) -> Stats:
    # Chat/RAG counters from the gateway, voice counters merged in from the
    # engine, so one snapshot covers the whole service.
    return gateway.stats().model_copy(update=tts.stats_summary())


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


async def _speak(tts: TTSEngine, text: str, voice_id: str | None, headers: dict | None = None):
    """Build a streaming audio response for ``text``.

    The first chunk is pulled eagerly so an upstream failure (ElevenLabs 401 /
    429 / timeout) surfaces here as the right HTTP status, BEFORE the 200 headers
    are sent — once streaming starts the status line can't be changed.
    """
    agen = tts.synthesize(text, voice_id)
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        first = b""
    except TTSError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))

    async def body():
        if first:
            yield first
        try:
            async for chunk in agen:
                yield chunk
        except TTSError:
            # Mid-stream failure after headers are already sent: stop cleanly
            # rather than corrupting the response. (Rare; the eager first-chunk
            # pull catches the common connect-time errors above.)
            return

    return StreamingResponse(body(), media_type=tts.content_type, headers=headers)


@app.post("/tts")
async def tts_synthesize(req: TTSRequest, tts: TTSEngine = Depends(get_tts)):
    """Turn text into speech. Streams audio/mpeg (small deterministic bytes on
    the offline mock)."""
    return await _speak(tts, req.text, req.voice_id)


@app.post("/rag/speak")
async def rag_speak(
    req: RagSpeakRequest,
    rag: RagPipeline = Depends(get_rag),
    tts: TTSEngine = Depends(get_tts),
    key: str = Depends(api_key),
):
    """RAG answers you can hear: run the normal RAG query, then stream the answer
    back as speech. Retrieval metadata (which doc chunks were used, and an ASCII
    preview of the answer) rides along in X-RAG-* response headers so a caller
    gets provenance without decoding the audio."""
    try:
        result = await rag.query(
            req.query, top_k=req.top_k, model=req.model, api_key=key
        )
    except RateLimitExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except AllProvidersFailed as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # HTTP headers must be latin-1; keep the preview ASCII-safe and bounded.
    preview = result.answer.encode("ascii", "ignore").decode()[:200]
    headers = {
        "X-RAG-Doc-Ids": ",".join(c.doc_id for c in result.chunks),
        "X-RAG-Chunks": str(len(result.chunks)),
        "X-RAG-Answer-Preview": preview,
    }
    return await _speak(tts, result.answer, req.voice_id, headers=headers)
