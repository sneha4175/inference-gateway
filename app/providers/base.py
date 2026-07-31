"""Provider abstraction.

The gateway must not care *which* LLM backend it is talking to. Every provider
(mock, OpenAI, a local model, ...) implements the same small interface, so the
router can swap or chain them freely. This is the classic "program to an
interface" idea and it is what makes provider fallback possible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.schemas import ChatRequest, ChatResponse


class ProviderError(Exception):
    """Raised when a provider fails (timeout, 5xx, bad key, ...).

    The router catches this to trigger fallback. Anything else propagates as a
    real bug rather than being silently swallowed.
    """


class Provider(ABC):
    """A backend that can answer a ChatRequest."""

    #: Human-readable name, surfaced in responses/logs (e.g. "mock", "openai").
    name: str = "base"

    @abstractmethod
    async def complete(self, req: ChatRequest) -> ChatResponse:
        """Return a full completion for the request."""

    @abstractmethod
    async def stream(self, req: ChatRequest) -> AsyncIterator[str]:
        """Yield the completion incrementally as text deltas."""
        raise NotImplementedError
