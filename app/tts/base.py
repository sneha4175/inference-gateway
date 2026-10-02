"""Text-to-speech provider abstraction.

The voice layer must not care *which* TTS backend produces the audio. Every
provider (mock, ElevenLabs, a local model, ...) implements the same small
interface, so the endpoints can swap one for another with a single env var. This
mirrors the chat `Provider` abstraction in app/providers/base.py — same "program
to an interface" idea, applied to speech.

Providers yield audio as a *stream* of byte chunks rather than one buffered blob,
so a long answer starts playing before the whole file is synthesised and memory
stays flat regardless of clip length.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class TTSError(Exception):
    """Raised when a TTS provider fails (timeout, 401, 429, 5xx, ...).

    Carries the HTTP status the endpoint should surface, so an upstream 429 stays
    a 429 to our caller and an auth/timeout failure maps to a sensible gateway
    code instead of a generic 500.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class TTSProvider(ABC):
    """A backend that can turn text into speech audio."""

    #: Human-readable name, surfaced in logs (e.g. "mock-tts", "elevenlabs").
    name: str = "base"
    #: MIME type of the audio this provider emits (ElevenLabs streams MP3).
    content_type: str = "audio/mpeg"

    @abstractmethod
    async def synthesize(
        self, text: str, voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        """Yield the synthesised speech for ``text`` as a stream of byte chunks.

        ``voice_id`` overrides the provider's configured default voice when given.
        """
        raise NotImplementedError
