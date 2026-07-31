"""End-to-end HTTP tests through the real FastAPI app (offline mock stack).

Uses Starlette's TestClient, which drives the ASGI app in-process — no server,
no network. This exercises routing, request/response models and DI wiring.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_completion_endpoint():
    r = client.post(
        "/v1/chat/completions",
        json={"model": "mock-1", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert "ping" in body["content"]
    assert body["cached"] is False
    assert body["usage"]["cost_usd"] == 0.0


def test_chat_streaming_endpoint():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-1",
            "messages": [{"role": "user", "content": "stream me"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert "stream" in r.text


def test_rag_ingest_then_query_endpoints():
    ingest = client.post(
        "/rag/ingest",
        json={
            "documents": ["RAG stands for retrieval augmented generation."],
            "doc_ids": ["rag-doc"],
        },
    )
    assert ingest.status_code == 200
    assert ingest.json()["chunks_indexed"] >= 1

    query = client.post("/rag/query", json={"query": "What does RAG stand for?", "top_k": 1})
    assert query.status_code == 200
    body = query.json()
    assert body["chunks"][0]["doc_id"] == "rag-doc"
    assert "retrieval" in body["answer"].lower()
