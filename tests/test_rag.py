"""RAG unit tests: chunking, and retrieval returns the topically-right chunk."""

from app.gateway.cache import ResponseCache
from app.gateway.rate_limiter import RateLimiter
from app.gateway.router import Gateway
from app.providers.mock import MockProvider
from app.rag.chunk import chunk_text
from app.rag.embed import HashingEmbedder
from app.rag.pipeline import RagPipeline


def _pipeline():
    gateway = Gateway([MockProvider()], ResponseCache(), RateLimiter(600, 50))
    return RagPipeline(embedder=HashingEmbedder(), gateway=gateway)


def test_chunking_overlaps_and_covers_text():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) >= 2
    # Overlap: the tail of chunk 0 should reappear at the head of chunk 1.
    assert chunks[0].split()[-1] in chunks[1].split()


def test_retrieval_returns_topically_relevant_chunk():
    pipe = _pipeline()
    pipe.ingest(
        documents=[
            "The mitochondria is the powerhouse of the cell and makes ATP energy.",
            "Python is a programming language used for web development and scripting.",
            "The Eiffel Tower is a wrought iron landmark located in Paris France.",
        ],
        doc_ids=["biology", "python", "paris"],
    )
    scored = pipe.store.search(pipe.embedder.embed("Where is the Eiffel Tower?"), top_k=1)
    assert scored[0].doc_id == "paris"


async def test_rag_end_to_end_uses_retrieved_context():
    pipe = _pipeline()
    pipe.ingest(
        documents=["The capital of Australia is Canberra, not Sydney."],
        doc_ids=["geo"],
    )
    result = await pipe.query("What is the capital of Australia?", top_k=1)

    # Retrieval found the geo chunk...
    assert result.chunks[0].doc_id == "geo"
    # ...and the mock provider (which echoes the augmented prompt) proves the
    # retrieved fact was actually fed to generation.
    assert "Canberra" in result.answer
    assert result.usage.cost_usd == 0.0
