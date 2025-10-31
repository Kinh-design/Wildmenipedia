from fastapi.testclient import TestClient

from grokalternative.api import app
from grokalternative.rag import hybrid_answer as real_hybrid_answer


def test_hybrid_answer_per_claim_citations():
    class FakeKG:
        def neighbors(self, node_id: str, limit: int = 10, offset: int = 0):
            return [
                {
                    "s": "E:Q1",
                    "p": "instance of",
                    "o": "E:Q5",
                    "meta": {"object_label": "human", "rank": "preferred", "pred_code": "P31"},
                },
                {
                    "s": "E:Q1",
                    "p": "educated at",
                    "o": "E:Q2",
                    "meta": {"object_label": "Some University", "rank": "normal"},
                },
            ]

    class FakeVS:
        def search(self, collection: str, vector: list[float], top_k: int = 5):
            return []

    web_docs = [
        {"url": "https://example.com/a", "title": "Human", "summary": "E:Q1 is a human", "text": "human"},
        {"url": "https://example.com/b", "title": "Education", "summary": "Some University", "text": "educated at Some University"},
    ]

    out = real_hybrid_answer("E:Q1", kg=FakeKG(), vs=FakeVS(), web_docs=web_docs)
    assert out.get("facts"), "expected facts from FakeKG"
    # Ensure per-claim top-1 citation and score fields are present
    f0 = out["facts"][0]
    assert isinstance(f0.get("citations"), list) and len(f0["citations"]) == 1
    assert isinstance(f0.get("top_score"), float)
    assert isinstance(f0.get("citations_all"), list) and 1 <= len(f0["citations_all"]) <= 5


def test_ui_renders_citations_and_export(monkeypatch):
    # Stub hybrid_answer to deterministic payload for rendering
    def stub_hybrid_answer(q: str, **kwargs):
        return {
            "answer": "Summary with [1]",
            "facts": [
                {
                    "subject": "E:Q1",
                    "predicate": "instance of",
                    "object": "E:Q5",
                    "meta": {"object_label": "human", "rank": "preferred", "pred_code": "P31"},
                    "score": 0.9,
                    "citations": [1],
                    "top_score": 0.88,
                    "citations_all": [{"n": 1, "score": 0.88}],
                }
            ],
            "sources": [{"url": "https://example.com/a", "title": "Human", "engine": "httpx"}],
            "vector_hits": [],
            "selected_entities": [q],
            "confidence": "high",
        }

    monkeypatch.setattr("grokalternative.api.hybrid_answer", stub_hybrid_answer)

    client = TestClient(app)
    r = client.get("/search", params={"q": "E:Q1"})
    assert r.status_code == 200
    html = r.text
    assert "href=\"#src-1\"" in html  # footnote link
    assert ">rel " in html  # similarity chip

    # Markdown export endpoint
    r2 = client.get("/export/markdown", params={"q": "E:Q1"})
    assert r2.status_code == 200
    assert r2.headers.get("content-type", "").startswith("text/markdown")
    md = r2.text
    assert "## Supporting facts" in md
