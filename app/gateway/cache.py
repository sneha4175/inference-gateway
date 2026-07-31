"""Response cache: skip the provider when we have already answered a request.

Why cache LLM responses: identical prompts are common (retries, shared prompts,
deterministic pipelines), and every avoided call saves latency AND money. Caching
is often the single biggest cost lever on an LLM gateway.

Design: an in-memory dict keyed by a hash of (model + messages), with a TTL and
LRU eviction. In-memory keeps it dependency-free for this project; production
would use Redis so the cache is shared across processes (see README trade-offs).

Caveat: caching only makes sense for *deterministic* requests. Real calls with
temperature > 0 are random, so a production gateway would make caching opt-in or
key on the sampling params too.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict

from app.schemas import ChatRequest, ChatResponse


def cache_key(req: ChatRequest) -> str:
    """Stable hash of the semantically-relevant parts of a request."""
    payload = {
        "model": req.model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000) -> None:
        self.ttl = ttl_seconds
        self.max_size = max_size
        # OrderedDict gives us cheap LRU: move_to_end on access, popitem(last=False)
        # to evict the oldest.
        self._store: "OrderedDict[str, tuple[float, ChatResponse]]" = OrderedDict()

    def get(self, key: str, now: float | None = None) -> ChatResponse | None:
        now = time.monotonic() if now is None else now
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, resp = item
        if now >= expires_at:
            # Lazily drop expired entries on read.
            del self._store[key]
            return None
        self._store.move_to_end(key)  # mark as recently used
        return resp

    def set(self, key: str, resp: ChatResponse, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._store[key] = (now + self.ttl, resp)
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)  # evict least-recently-used
