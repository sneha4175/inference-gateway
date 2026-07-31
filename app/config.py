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

    # Default offline stack: primary mock + a second mock for a visible chain.
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
