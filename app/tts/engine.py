"""The voice engine: wraps a TTS provider with the gateway's observability.

Why a thin wrapper rather than calling the provider directly from the route:
counting requests and timing latency belongs in one place, the same way the chat
Gateway owns those concerns for completions. The engine reuses the gateway's own
``LatencyWindow`` so /stats and /metrics report TTS timing exactly like chat
timing (single mechanism, no second implementation to drift).

Latency is measured as time-to-first-byte: the clock starts when synthesis is
requested and is recorded when the first audio chunk arrives. For a streaming
API that is the number that matters — how long until audio can start playing.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from app.gateway.router import LatencyWindow
from app.tts.base import TTSProvider


class TTSEngine:
    def __init__(self, provider: TTSProvider) -> None:
        self.provider = provider
        self.latencies = LatencyWindow()
        self._requests = 0

    @property
    def content_type(self) -> str:
        return self.provider.content_type

    @property
    def count(self) -> int:
        return self._requests

    async def synthesize(
        self, text: str, voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Stream audio for ``text``, recording the request and its TTFB latency.

        A ``TTSError`` from the provider propagates unchanged so the route can map
        it to the right HTTP status.
        """
        self._requests += 1
        start = time.perf_counter()
        recorded = False
        async for chunk in self.provider.synthesize(text, voice_id):
            if not recorded:
                self.latencies.record((time.perf_counter() - start) * 1000.0)
                recorded = True
            yield chunk
        if not recorded:
            # Provider yielded nothing; still record the (small) elapsed time so
            # the request is represented in the latency window.
            self.latencies.record((time.perf_counter() - start) * 1000.0)

    def stats_summary(self) -> dict[str, float]:
        """TTS counters for /stats, namespaced so they sit beside the chat ones."""
        lat = self.latencies.summary()
        return {
            "tts_requests": self._requests,
            "tts_latency_ms_p50": lat["latency_ms_p50"],
            "tts_latency_ms_p95": lat["latency_ms_p95"],
            "tts_latency_ms_avg": lat["latency_ms_avg"],
        }
