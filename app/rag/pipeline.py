"""The RAG pipeline, exposed through the same gateway.

Ingest:   documents -> chunk -> embed -> store
Query:    question -> embed -> retrieve top-k -> augment prompt -> generate

The key design point: retrieval produces *context*, and generation goes through
the SAME Gateway used by /v1/chat/completions. So RAG answers inherit rate
limiting, caching, cost tracking and provider fallback for free — the whole
reason to put RAG "behind the gateway" instead of bolting on a second path.
"""

from __future__ import annotations

from app.gateway.router import Gateway
from app.rag.chunk import chunk_text
from app.rag.embed import Embedder
from app.rag.store import VectorStore
from app.schemas import (
    ChatRequest,
    IngestResponse,
    Message,
    RagQueryResponse,
    RetrievedChunk,
)

# Instruction that tells the model to stay grounded in retrieved context. Putting
# it in a system message is a light guardrail against the model answering from
# parametric memory (and a small mitigation for prompt injection — see README).
_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the context does not contain the answer, say you don't know."
)


class RagPipeline:
    def __init__(self, embedder: Embedder, gateway: Gateway) -> None:
        self.embedder = embedder
        self.gateway = gateway
        self.store = VectorStore(dim=embedder.dim)

    def ingest(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
        *,
        chunk_size: int = 40,
        overlap: int = 10,
    ) -> IngestResponse:
        if doc_ids is not None and len(doc_ids) != len(documents):
            raise ValueError("doc_ids must have one id per document")
        ids = doc_ids or [f"doc-{i}" for i in range(len(documents))]

        all_chunks: list[str] = []
        all_ids: list[str] = []
        for doc, doc_id in zip(documents, ids):
            pieces = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(pieces)
            all_ids.extend([doc_id] * len(pieces))

        if all_chunks:
            vectors = self.embedder.embed_many(all_chunks)
            self.store.add(vectors, all_chunks, all_ids)

        return IngestResponse(
            chunks_indexed=len(all_chunks), total_chunks=len(self.store)
        )

    def _build_messages(self, query: str, chunks: list[RetrievedChunk]) -> list[Message]:
        context = "\n\n".join(f"[{c.doc_id}] {c.text}" for c in chunks)
        return [
            Message(role="system", content=_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Context:\n{context}\n\nQuestion: {query}",
            ),
        ]

    async def query(
        self, query: str, top_k: int = 3, model: str = "mock-1", api_key: str = "anonymous"
    ) -> RagQueryResponse:
        # 1) Retrieve.
        q_vec = self.embedder.embed(query)
        scored = self.store.search(q_vec, top_k=top_k)
        chunks = [
            RetrievedChunk(text=s.text, score=s.score, doc_id=s.doc_id) for s in scored
        ]

        # 2) Augment + 3) Generate (through the gateway).
        messages = self._build_messages(query, chunks)
        req = ChatRequest(model=model, messages=messages)
        resp = await self.gateway.chat(req, api_key=api_key)

        return RagQueryResponse(answer=resp.content, chunks=chunks, usage=resp.usage)
