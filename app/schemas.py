"""Pydantic models = the HTTP contract of the service.

Why pydantic at the boundary: it validates/parses untrusted request bodies and
serialises responses. Internally we pass these same objects around so there is
a single source of truth for shapes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant"]


# --- Chat / gateway models ---------------------------------------------------
class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    # Default model is the offline mock so the service works with zero config.
    model: str = "mock-1"
    messages: list[Message]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    # Wall-clock time the gateway spent serving THIS request, in milliseconds.
    # Set by the gateway (providers leave it at the 0.0 default); a cache hit
    # records a genuinely smaller value than the provider call it skipped.
    latency_ms: float = 0.0


class ChatResponse(BaseModel):
    id: str
    model: str
    provider: str
    content: str
    usage: Usage
    # True when served from the response cache instead of a provider call.
    cached: bool = False


# --- RAG models --------------------------------------------------------------
class IngestRequest(BaseModel):
    documents: list[str]
    # Optional parallel list of ids; one id per document. Auto-generated if None.
    doc_ids: list[str] | None = None


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_chunks: int


class RetrievedChunk(BaseModel):
    text: str
    score: float
    doc_id: str


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)
    model: str = "mock-1"


class RagQueryResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
    usage: Usage


# --- TTS (voice) models ------------------------------------------------------
class TTSRequest(BaseModel):
    text: str = Field(min_length=1)
    # Override the configured default voice for this one request (optional).
    voice_id: str | None = None


class RagSpeakRequest(RagQueryRequest):
    """A RAG query whose answer is returned as speech. Same fields as a normal
    RAG query, plus an optional per-request voice override."""

    voice_id: str | None = None


# --- Ops models --------------------------------------------------------------
class Stats(BaseModel):
    requests: int
    cache_hits: int
    cache_misses: int
    provider_errors: int
    rate_limit_rejections: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    # Latency percentiles over a bounded rolling window of recent requests (ms).
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_avg: float
    # Voice layer: synthesis request count + latency over its own rolling window.
    # Default 0 so a gateway built without a TTS engine still serialises cleanly.
    tts_requests: int = 0
    tts_latency_ms_p50: float = 0.0
    tts_latency_ms_p95: float = 0.0
    tts_latency_ms_avg: float = 0.0
