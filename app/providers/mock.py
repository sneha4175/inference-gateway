"""Deterministic mock provider.

This is the single most important piece for "runs fully offline". It implements
the exact same interface as a real provider but never touches the network, so:
  * the whole service (incl. RAG) works with no API key and $0 cost,
  * tests are deterministic — same input always yields the same output,
  * fallback can be exercised by pointing the router at a mock that always fails.

Behaviour: it echoes the last user message back. That sounds trivial, but it is
exactly what makes RAG verifiable offline — the RAG pipeline stuffs the
retrieved context into that user message, so if retrieval worked, the retrieved
text shows up in the mock's answer and a test can assert on it.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

from app.gateway.cost import build_usage, prompt_tokens
from app.providers.base import Provider, ProviderError
from app.schemas import ChatRequest, ChatResponse


def _last_user_message(req: ChatRequest) -> str:
    for m in reversed(req.messages):
        if m.role == "user":
            return m.content
    return ""


class MockProvider(Provider):
    def __init__(self, name: str = "mock", *, should_fail: bool = False) -> None:
        self.name = name
        # When True, every call raises ProviderError. Used to simulate an outage
        # so the router's fallback path can be tested.
        self.should_fail = should_fail

    def _reply_text(self, req: ChatRequest) -> str:
        user = _last_user_message(req).strip()
        return f"[{self.name}:{req.model}] {user}"

    def _response(self, req: ChatRequest, text: str) -> ChatResponse:
        p_toks = prompt_tokens(req.messages)
        # Deterministic id derived from the prompt so identical calls look
        # identical (handy when eyeballing cache behaviour).
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        return ChatResponse(
            id=f"mock-{digest}",
            model=req.model,
            provider=self.name,
            content=text,
            usage=build_usage(req.model, p_toks, text),
        )

    async def complete(self, req: ChatRequest) -> ChatResponse:
        if self.should_fail:
            raise ProviderError(f"{self.name}: simulated provider outage")
        return self._response(req, self._reply_text(req))

    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        if self.should_fail:
            raise ProviderError(f"{self.name}: simulated provider outage")
        # Yield word-by-word to mimic token streaming from a real provider.
        for word in self._reply_text(req).split(" "):
            yield word + " "
