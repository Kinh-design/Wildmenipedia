from fastapi.testclient import TestClient

from grokalternative.api import app


def test_ask_endpoint():
    client = TestClient(app)
    r = client.get("/ask", params={"q": "quantum entanglement"})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "answer" in data
    assert isinstance(data.get("citations", []), list)
