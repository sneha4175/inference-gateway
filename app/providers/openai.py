"""Real, OpenAI-compatible provider (pluggable, not exercised in offline tests).

Why "OpenAI-compatible" rather than "OpenAI": the /chat/completions schema is a
de-facto standard. OpenAI, Together, Groq, OpenRouter, vLLM and Ollama all speak
it. By targeting the wire format (and letting `base_url` be overridden) one class
covers many backends — no per-vendor SDK.

This is intentionally thin and is only used when PROVIDER=openai and a key is
set. The offline test suite never imports it with a real key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from app.gateway.cost import build_usage, prompt_tokens
from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, ChatResponse


class OpenAICompatibleProvider(Provider):
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        name: str = "openai",
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _payload(self, req: ChatRequest, stream: bool) -> dict:
        return {
            "model": req.model,
            "messages": [m.model_dump() for m in req.messages],
            "stream": stream,
            **({"max_tokens": req.max_tokens} if req.max_tokens else {}),
        }

    async def complete(self, req: ChatRequest) -> ChatResponse:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(req, stream=False),
                )
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Wrap every transport/parse failure so the router can fall back.
            raise ProviderError(f"{self.name}: {exc}") from exc

        text = data["choices"][0]["message"]["content"]
        # Prefer the provider's own token accounting when present; otherwise
        # fall back to our approximate counter so cost is never missing.
        usage = data.get("usage")
        if usage:
            from app.gateway.cost import estimate_cost
            from app.schemas import Usage

            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            usage_model = Usage(
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=usage.get("total_tokens", pt + ct),
                cost_usd=estimate_cost(req.model, pt, ct),
            )
        else:
            usage_model = build_usage(req.model, prompt_tokens(req.messages), text)

        return ChatResponse(
            id=data.get("id", "openai-unknown"),
            model=data.get("model", req.model),
            provider=self.name,
            content=text,
            usage=usage_model,
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=self._payload(req, stream=True),
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        # OpenAI streams Server-Sent Events: "data: {json}".
                        if not line.startswith("data: "):
                            continue
                        chunk = line[len("data: "):]
                        if chunk.strip() == "[DONE]":
                            break
                        import json

                        delta = json.loads(chunk)["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc
