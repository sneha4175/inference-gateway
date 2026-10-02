"""Voice-layer tests, fully offline (mock TTS provider, no key, no network).

Covers the deterministic mock, config wiring, the /tts and /rag/speak HTTP
routes, and the TTS counters surfaced on /stats and /metrics.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, build_tts
from app.main import app
from app.tts.base import TTSError
from app.tts.elevenlabs import ElevenLabsTTSProvider
from app.tts.engine import TTSEngine
from app.tts.mock import MockTTSProvider

client = TestClient(app)


async def _collect(provider, text, voice_id=None):
    return b"".join([chunk async for chunk in provider.synthesize(text, voice_id)])


# --- mock provider -----------------------------------------------------------
async def test_mock_tts_is_deterministic_and_nonempty():
    a = await _collect(MockTTSProvider(), "hello voice")
    b = await _collect(MockTTSProvider(), "hello voice")
    assert a == b              # same input -> identical bytes
    assert len(a) > 0


async def test_mock_tts_longer_text_yields_more_audio():
    short = await _collect(MockTTSProvider(), "hi")
    long = await _collect(MockTTSProvider(), "hi " * 200)
    assert len(long) > len(short)


async def test_mock_tts_voice_id_changes_output():
    rachel = await _collect(MockTTSProvider(), "same text", "voice-a")
    adam = await _collect(MockTTSProvider(), "same text", "voice-b")
    assert rachel != adam


async def test_mock_tts_failure_raises_tts_error():
    with pytest.raises(TTSError):
        await _collect(MockTTSProvider(should_fail=True), "boom")


# --- config ------------------------------------------------------------------
def test_tts_defaults_to_mock():
    assert Settings().tts_provider == "mock"
    assert isinstance(build_tts(Settings()).provider, MockTTSProvider)


def test_elevenlabs_without_key_raises_clear_error():
    settings = Settings(tts_provider="elevenlabs", elevenlabs_api_key=None)
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY"):
        build_tts(settings)


def test_unknown_tts_provider_raises():
    with pytest.raises(RuntimeError, match="unknown TTS_PROVIDER"):
        build_tts(Settings(tts_provider="espeak"))


# --- /tts endpoint -----------------------------------------------------------
def test_tts_endpoint_returns_audio():
    r = client.post("/tts", json={"text": "read this aloud"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert len(r.content) > 0


def test_tts_endpoint_rejects_empty_text():
    assert client.post("/tts", json={"text": ""}).status_code == 422


# --- /rag/speak endpoint -----------------------------------------------------
def test_rag_speak_returns_spoken_answer_after_retrieval():
    ingest = client.post(
        "/rag/ingest",
        json={
            "documents": ["The capital of Australia is Canberra, not Sydney."],
            "doc_ids": ["geo-speak"],
        },
    )
    assert ingest.status_code == 200

    r = client.post(
        "/rag/speak",
        json={"query": "What is the capital of Australia?", "top_k": 1},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    # Audio was produced for the answer.
    assert len(r.content) > 0
    # Retrieval actually fired: the geo doc was the retrieved chunk...
    assert r.headers["X-RAG-Doc-Ids"] == "geo-speak"
    # ...and the retrieved fact reached the (spoken) answer.
    assert "Canberra" in r.headers["X-RAG-Answer-Preview"]


# --- observability -----------------------------------------------------------
async def test_engine_tracks_requests_and_latency():
    engine = TTSEngine(MockTTSProvider())
    await _collect(engine, "one")
    await _collect(engine, "two")
    summary = engine.stats_summary()
    assert summary["tts_requests"] == 2
    assert summary["tts_latency_ms_avg"] >= 0.0


def test_stats_and_metrics_expose_tts_counters():
    client.post("/tts", json={"text": "count me"})

    stats = client.get("/stats").json()
    assert "tts_requests" in stats
    assert stats["tts_requests"] >= 1
    for field in ("tts_latency_ms_p50", "tts_latency_ms_p95", "tts_latency_ms_avg"):
        assert field in stats

    metrics = client.get("/metrics").text
    assert "gateway_tts_requests_total" in metrics
    assert "gateway_tts_latency_ms_count" in metrics
    assert "gateway_tts_latency_ms_p50" in metrics


# --- ElevenLabs real path (offline: a fake transport, never the network) -----
def _provider_with_transport(handler, **kw):
    """ElevenLabs provider whose httpx client uses an in-memory MockTransport, so
    the real request-building code runs with zero network."""
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _client(**client_kw):
        client_kw.pop("transport", None)
        return real_client(transport=transport, **client_kw)

    provider = ElevenLabsTTSProvider(
        "secret-key", voice_id="voice-xyz", model="eleven_turbo_v2_5", **kw
    )
    return provider, _client


async def _collect_patched(provider, client_factory, monkeypatch, text, voice_id=None):
    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    return b"".join([c async for c in provider.synthesize(text, voice_id)])


async def test_elevenlabs_builds_correct_streaming_request(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["key"] = request.headers.get("xi-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, content=b"\x00\x01REALMP3")

    provider, factory = _provider_with_transport(handler)
    audio = await _collect_patched(provider, factory, monkeypatch, "speak this")

    assert audio == b"\x00\x01REALMP3"
    assert captured["method"] == "POST"
    # Hits the documented streaming endpoint for the configured voice.
    assert captured["url"].endswith("/text-to-speech/voice-xyz/stream")
    assert captured["key"] == "secret-key"
    assert captured["body"]["text"] == "speak this"
    assert captured["body"]["model_id"] == "eleven_turbo_v2_5"


async def test_elevenlabs_per_request_voice_override(monkeypatch):
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, content=b"ok")

    provider, factory = _provider_with_transport(handler)
    await _collect_patched(provider, factory, monkeypatch, "hi", voice_id="override-voice")
    assert captured["url"].endswith("/text-to-speech/override-voice/stream")


@pytest.mark.parametrize("code,expected", [(401, 502), (429, 429), (500, 502)])
async def test_elevenlabs_maps_upstream_errors(monkeypatch, code, expected):
    provider, factory = _provider_with_transport(
        lambda req: httpx.Response(code, content=b"err")
    )
    monkeypatch.setattr(httpx, "AsyncClient", factory)
    with pytest.raises(TTSError) as ei:
        async for _ in provider.synthesize("boom"):
            pass
    assert ei.value.status_code == expected
