"""Chunking: split a document into retrievable pieces.

Why chunk at all: you cannot embed and retrieve a whole 50-page document as one
vector — retrieval would be coarse and the chunk would blow the model's context
window. Splitting into ~paragraph-sized pieces lets retrieval pull only the few
passages relevant to a question.

Why the overlap: a naive hard split can cut a sentence (and its meaning) across
two chunks, so a query might match neither well. A small overlap (sliding window)
keeps boundary context in both neighbours. Chunk size / overlap are the classic
RAG tuning knobs.

This is a simple word-count splitter. Production systems often split on sentence
or markdown structure; word count is transparent and dependency-free, which is
the right trade-off here.
"""

from __future__ import annotations

from app.util import tokenize


def chunk_text(text: str, chunk_size: int = 40, overlap: int = 10) -> list[str]:
    """Split `text` into overlapping chunks of ~chunk_size tokens.

    chunk_size/overlap are in tokens (see app.util.tokenize). Returns the chunks
    as reconstructed strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenize(text)
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break  # last window reached the end
    return chunks
