"""Embedders: turn text into a vector so we can compare meaning by geometry.

Two implementations behind one interface:

* HashingEmbedder (default, offline): the "hashing trick" / feature hashing.
  Each token is hashed to a bucket in a fixed-size vector. No model, no network,
  no downloads — fully deterministic. It captures *lexical overlap*: texts that
  share words get similar vectors, which is enough to make retrieval demonstrably
  work in tests. It does NOT capture semantics (synonyms), and that's an honest
  limitation, not a bug — see README.

* OpenAIEmbedder (pluggable): calls a real embedding API for semantic vectors.
  Only used when EMBEDDER=openai + a key is set; never in offline tests.

Both return L2-normalised vectors, so a dot product == cosine similarity.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from app.util import tokenize


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        ...

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Embed a batch into a (n, dim) matrix."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            # First 4 bytes pick the bucket; the 5th byte picks a sign. The sign
            # trick reduces collisions cancelling into systematic bias.
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        return _normalize(vec)


class OpenAIEmbedder(Embedder):
    """Real embeddings via an OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
    ) -> None:
        self.dim = dim
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")

    def embed(self, text: str) -> np.ndarray:
        import httpx

        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
        return _normalize(vec)
