from __future__ import annotations

from typing import Dict

import httpx
from fastapi.testclient import TestClient

from grokalternative import llm as llm_mod
from grokalternative.api import app


def test_sse_stream_basic():
    client = TestClient(app)
    # Seed cookie/session
    client.get("/chat")

    with client.stream("GET", "/chat/stream", params={"q": "Hello world"}) as r:
        r.raise_for_status()
        total = ""
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                try:
                    import json

                    d = json.loads(payload)
                    total += d.get("delta", "")
                except Exception:
                    pass
        assert "synthesized" in total or total.strip() != ""


def test_llm_request_with_retries(monkeypatch):
    calls: Dict[str, int] = {"n": 0}

    def fake_request(self: httpx.Client, method: str, url: str, headers=None, json=None, timeout=30.0):  # type: ignore[no-redef]
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.HTTPError("temporary failure")
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx.Client, "request", fake_request, raising=True)

    with httpx.Client() as client:
        resp = llm_mod._request_with_retries(client, "GET", "https://example.com/ok", retries=2, backoff=0.01)  # noqa: SLF001
    assert resp.status_code == 200
    assert calls["n"] == 2
