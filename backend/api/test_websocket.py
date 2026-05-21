from __future__ import annotations

from starlette.testclient import TestClient

from main import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_websocket_echo() -> None:
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello", "payload": {"message": "hello from frontend"}})
        data = ws.receive_json()
        assert data["type"] == "hello_ack"
        assert data["payload"]["echo"] == "hello from frontend"
        assert isinstance(data["payload"]["server_timestamp_ms"], int)
        assert data["payload"]["server_timestamp_ms"] > 0
