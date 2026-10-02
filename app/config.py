"""Runtime configuration + object wiring (the composition root).

Everything pluggable is chosen here from environment variables, so the rest of
the code never reads os.environ and never hard-codes a provider. Defaults are the
OFFLINE MOCK stack, so `uvicorn app.main:app` just works with zero config.

We read os.environ directly rather than pulling in pydantic-settings — one fewer
dependency for a handful of values (New-Thing checklist: not worth a library).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.gateway.cache import ResponseCache
from app.gateway.rate_limiter import RateLimiter
from app.gateway.router import Gateway
from app.providers.base import Provider
from app.providers.mock import MockProvider
from app.rag.embed import Embedder, HashingEmbedder
from app.rag.pipeline import RagPipeline
from app.tts.engine import TTSEngine
from app.tts.mock import MockTTSProvider


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    provider: str = "mock"          # mock | openai
    embedder: str = "hashing"       # hashing | openai
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    rate_limit_per_min: int = 60
    rate_limit_burst: int = 10
    cache_ttl_seconds: int = 300
    # --- Text-to-speech (voice layer) ---
    tts_provider: str = "mock"      # mock | elevenlabs
    elevenlabs_api_key: str | None = None
    # Default voice "Rachel" — a stock ElevenLabs voice present on every account,
    # so the real path works out of the box with just a key.
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_turbo_v2_5"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=_env("PROVIDER", "mock"),
            embedder=_env("EMBEDDER", "hashing"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            rate_limit_per_min=int(_env("RATE_LIMIT_PER_MIN", "60")),
            rate_limit_burst=int(_env("RATE_LIMIT_BURST", "10")),
            cache_ttl_seconds=int(_env("CACHE_TTL_SECONDS", "300")),
            tts_provider=_env("TTS_PROVIDER", "mock"),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            elevenlabs_model=_env("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
        )


def build_providers(settings: Settings) -> list[Provider]:
    """Build the fallback chain. Primary first, then a mock safety net.

    Putting a MockProvider LAST means the gateway degrades to a canned response
    instead of a hard 502 if the real provider is down. Whether you want that in
    production is a policy call; here it also keeps a real-provider deployment
    demoable without a second paid vendor.
    """
    if settings.provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("PROVIDER=openai but OPENAI_API_KEY is not set")
        # Imported lazily so httpx-based real provider isn't required offline.
        from app.providers.openai import OpenAICompatibleProvider

        primary = OpenAICompatibleProvider(
            settings.openai_api_key, base_url=settings.openai_base_url
        )
        return [primary, MockProvider(name="mock-fallback")]

    # Default offline stack: a single mock provider (a one-element chain). The
    # multi-provider fallback path is exercised with PROVIDER=openai and in
    # tests/test_fallback.py.
    return [MockProvider(name="mock")]


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("EMBEDDER=openai but OPENAI_API_KEY is not set")
        from app.rag.embed import OpenAIEmbedder

        return OpenAIEmbedder(settings.openai_api_key, base_url=settings.openai_base_url)
    return HashingEmbedder()


def build_gateway(settings: Settings) -> Gateway:
    return Gateway(
        providers=build_providers(settings),
        cache=ResponseCache(ttl_seconds=settings.cache_ttl_seconds),
        limiter=RateLimiter(
            rate_per_min=settings.rate_limit_per_min, burst=settings.rate_limit_burst
        ),
    )


def build_rag(settings: Settings, gateway: Gateway) -> RagPipeline:
    return RagPipeline(embedder=build_embedder(settings), gateway=gateway)


def build_tts(settings: Settings) -> TTSEngine:
    """Build the voice engine. Default is the offline mock (no key, no network).

    Only the elevenlabs path needs a key; selecting it without one is a hard
    config error rather than a silent fallback, so a misconfigured deployment
    fails fast instead of quietly going mute.
    """
    if settings.tts_provider == "elevenlabs":
        if not settings.elevenlabs_api_key:
            raise RuntimeError(
                "TTS_PROVIDER=elevenlabs but ELEVENLABS_API_KEY is not set"
            )
        # Imported lazily so the httpx-based real provider isn't needed offline.
        from app.tts.elevenlabs import ElevenLabsTTSProvider

        provider = ElevenLabsTTSProvider(
            settings.elevenlabs_api_key,
            voice_id=settings.elevenlabs_voice_id,
            model=settings.elevenlabs_model,
        )
        return TTSEngine(provider)

    if settings.tts_provider == "mock":
        return TTSEngine(MockTTSProvider())

    raise RuntimeError(
        f"unknown TTS_PROVIDER={settings.tts_provider!r} (expected 'mock' or 'elevenlabs')"
    )
