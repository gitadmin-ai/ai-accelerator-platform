import pytest
from fastapi.testclient import TestClient

from backend.api.main import create_app
from backend.settings_store import SettingsStore


@pytest.fixture
def client():
    app = create_app(settings_store=SettingsStore(":memory:"))
    return TestClient(app)


def test_get_settings_returns_defaults(client):
    resp = client.get("/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_storage_backend"] == "local"
    assert body["dataset_storage_backend"] == "local"
    assert body["checkpoint_storage_backend"] == "local"
    assert body["ddl_server"] is None


def test_put_settings_updates_fields(client):
    resp = client.put("/settings", json={"ddl_server": "nebula.internal", "checkpoint_storage_backend": "ddl"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ddl_server"] == "nebula.internal"
    assert body["checkpoint_storage_backend"] == "ddl"

    refetched = client.get("/settings").json()
    assert refetched == body


def test_put_settings_rejects_ddl_without_server(client):
    resp = client.put("/settings", json={"model_storage_backend": "ddl"})
    assert resp.status_code == 400


def test_put_settings_rejects_unknown_backend_value(client):
    resp = client.put("/settings", json={"dataset_storage_backend": "s3"})
    assert resp.status_code == 400
