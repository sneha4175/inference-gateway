"""Real ElevenLabs TTS provider (pluggable, not exercised in offline tests).

Thin wrapper over the ElevenLabs streaming TTS endpoint:

    POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream

authenticated with an ``xi-api-key`` header. It streams MP3 back, which we hand
straight through to the client so playback can start before synthesis finishes.

Like the OpenAI-compatible chat provider, this is intentionally thin and is only
used when TTS_PROVIDER=elevenlabs with a key set. The offline test suite never
imports it with a real key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.tts.base import TTSError, TTSProvider


class ElevenLabsTTSProvider(TTSProvider):
    name = "elevenlabs"
    content_type = "audio/mpeg"

    def __init__(
        self,
        api_key: str,
        *,
        voice_id: str,
        model: str,
        base_url: str = "https://api.elevenlabs.io/v1",
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"xi-api-key": self._api_key, "accept": "audio/mpeg"}

    def _payload(self, text: str) -> dict:
        return {
            "text": text,
            "model_id": self._model,
            # Middle-of-the-road defaults; tune per voice in a real deployment.
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

    async def synthesize(
        self, text: str, voice_id: str | None = None
    ) -> AsyncIterator[bytes]:
        vid = voice_id or self._voice_id
        url = f"{self._base_url}/text-to-speech/{vid}/stream"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", url, headers=self._headers(), json=self._payload(text)
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
        except httpx.TimeoutException as exc:
            raise TTSError(f"elevenlabs: request timed out ({exc})", status_code=504) from exc
        except httpx.HTTPStatusError as exc:
            raise self._status_error(exc) from exc
        except httpx.HTTPError as exc:
            raise TTSError(f"elevenlabs: {exc}", status_code=502) from exc

    @staticmethod
    def _status_error(exc: httpx.HTTPStatusError) -> TTSError:
        """Map an upstream HTTP error to a clear, actionable gateway error."""
        code = exc.response.status_code
        if code == 401:
            return TTSError(
                "elevenlabs: rejected the API key (401) — check ELEVENLABS_API_KEY",
                status_code=502,
            )
        if code == 429:
            return TTSError(
                "elevenlabs: rate limit / quota exceeded (429)", status_code=429
            )
        return TTSError(f"elevenlabs: upstream returned HTTP {code}", status_code=502)
