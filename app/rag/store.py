"""A tiny in-memory vector store backed by NumPy.

Why not FAISS / a vector DB: for a portfolio-scale corpus, brute-force cosine
similarity over a NumPy matrix is exact, ~20 lines, and has no heavy deps. FAISS
and pgvector/Pinecone matter at millions of vectors where an approximate index
beats a linear scan — over-kill here and a distraction from the concepts. This is
the deliberate "minimal deps" choice from the brief. See README trade-offs.

Vectors are stored pre-normalised (the embedders do this), so cosine similarity
is just a matrix-vector dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoredChunk:
    text: str
    doc_id: str
    score: float


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._matrix = np.empty((0, dim), dtype=np.float32)
        self._texts: list[str] = []
        self._doc_ids: list[str] = []

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, vectors: np.ndarray, texts: list[str], doc_ids: list[str]) -> None:
        if len(texts) != len(doc_ids) or len(texts) != vectors.shape[0]:
            raise ValueError("vectors, texts and doc_ids must line up")
        if vectors.shape[0] == 0:
            return
        self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])
        self._texts.extend(texts)
        self._doc_ids.extend(doc_ids)

    def search(self, query_vec: np.ndarray, top_k: int = 3) -> list[ScoredChunk]:
        """Return the top_k most similar chunks (highest cosine first)."""
        if len(self) == 0:
            return []
        # Dot product against every stored (normalised) vector == cosine sim.
        scores = self._matrix @ query_vec.astype(np.float32)
        k = min(top_k, len(self))
        # argpartition gets the top-k cheaply, then we sort just those k.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            ScoredChunk(
                text=self._texts[i],
                doc_id=self._doc_ids[i],
                score=float(scores[i]),
            )
            for i in top_idx
        ]

    def clear(self) -> None:
        self._matrix = np.empty((0, self.dim), dtype=np.float32)
        self._texts.clear()
        self._doc_ids.clear()
