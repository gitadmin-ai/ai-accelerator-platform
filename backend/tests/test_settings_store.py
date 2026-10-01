import pytest

from backend.settings_store import SettingsStore, SettingsValidationError


def test_get_creates_default_row_on_first_call():
    store = SettingsStore(":memory:")
    settings = store.get()
    assert settings["model_storage_backend"] == "local"
    assert settings["dataset_storage_backend"] == "local"
    assert settings["checkpoint_storage_backend"] == "local"
    assert settings["ddl_server"] is None
    assert settings["ddl_port"] == 58000
    assert settings["ddl_tenant"] == 1
    assert settings["ddl_cpu_base"] == 0


def test_get_is_idempotent():
    store = SettingsStore(":memory:")
    first = store.get()
    second = store.get()
    assert first == second


def test_update_persists_fields():
    store = SettingsStore(":memory:")
    updated = store.update(ddl_server="nebula.internal", ddl_port=59000, checkpoint_storage_backend="ddl")
    assert updated["ddl_server"] == "nebula.internal"
    assert updated["ddl_port"] == 59000
    assert updated["checkpoint_storage_backend"] == "ddl"
    # unrelated fields untouched
    assert updated["model_storage_backend"] == "local"

    reread = store.get()
    assert reread == updated


def test_update_rejects_ddl_backend_without_server():
    store = SettingsStore(":memory:")
    with pytest.raises(SettingsValidationError):
        store.update(model_storage_backend="ddl")


def test_update_rejects_unknown_backend_value():
    store = SettingsStore(":memory:")
    with pytest.raises(SettingsValidationError):
        store.update(dataset_storage_backend="s3")


def test_setting_ddl_server_and_backend_together_is_allowed():
    store = SettingsStore(":memory:")
    updated = store.update(model_storage_backend="ddl", ddl_server="nebula.internal")
    assert updated["model_storage_backend"] == "ddl"
    assert updated["ddl_server"] == "nebula.internal"


def test_previously_set_ddl_server_satisfies_a_later_backend_switch():
    store = SettingsStore(":memory:")
    store.update(ddl_server="nebula.internal")
    updated = store.update(dataset_storage_backend="ddl")
    assert updated["dataset_storage_backend"] == "ddl"
