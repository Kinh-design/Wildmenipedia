from fastapi.testclient import TestClient

from grokalternative.api import app


def test_health_ok():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
