"""Deterministic mock TTS provider.

This is what keeps the voice layer "offline-first", exactly like MockProvider
does for chat. It implements the same interface as a real TTS backend but never
touches the network, so:
  * /tts and /rag/speak work with no ElevenLabs key and $0 cost,
  * tests are deterministic — the same text always yields the same bytes,
  * the audio length grows with the text length, so a test can tell a short clip
    from a long one without a real synthesiser.

The bytes are NOT playable audio; they are a stable, non-empty stand-in whose
content is a pure function of (voice_id, text). A readable ASCII header makes it
obvious in a hex dump that this came from the mock and not a real provider.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

from app.tts.base import TTSError, TTSProvider

# Prepended to every mock clip so it is self-identifying in a dump/log.
_HEADER = b"MOCKTTS1"
# Emitted chunk size, so the mock exercises the same streaming path as a real
# provider (the endpoint and tests see several chunks, not one blob).
_CHUNK = 256


def _fake_audio(text: str, voice_id: str) -> bytes:
    """Deterministic pseudo-audio whose length scales with the input text.

    One SHA-256 block (32 bytes) is repeated once per ~8 characters of input, so
    longer text produces a longer clip — a cheap, reproducible analogue of real
    synthesis that lets tests assert "longer text -> more audio".
    """
    seed = hashlib.sha256(f"{voice_id}\x00{text}".encode()).digest()
    repeats = max(1, len(text) // 8)
    return _HEADER + seed * repeats


class MockTTSProvider(TTSProvider):
    name = "mock-tts"
    content_type = "audio/mpeg"

    def __init__(self, name: str = "mock-tts", *, should_fail: bool = False) -> None:
        self.name = name
        # When True every call raises TTSError, to exercise the error path.
        self.should_fail = should_fail

    async def synthesize(
        self, text: str, voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        if self.should_fail:
            raise TTSError(f"{self.name}: simulated synthesis failure")
        audio = _fake_audio(text, voice_id or "default")
        for i in range(0, len(audio), _CHUNK):
            yield audio[i : i + _CHUNK]
