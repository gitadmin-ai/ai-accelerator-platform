from nebula_ddl_storage.connection import is_not_found_error, key_to_object_id


def test_key_to_object_id_is_deterministic():
    a = key_to_object_id("models/abc123/manifest.json")
    b = key_to_object_id("models/abc123/manifest.json")
    assert a == b


def test_key_to_object_id_differs_for_different_keys():
    a = key_to_object_id("models/abc123/manifest.json")
    b = key_to_object_id("datasets/abc123/manifest.json")
    assert a != b


def test_key_to_object_id_matches_known_vector():
    # Vendored verbatim from NebulaR's ddl_checkpoint_keys.py: first 16
    # bytes of SHA-256(key), split into two big-endian 64-bit halves.
    import hashlib

    key = "some/example/key"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    expected_hi = int.from_bytes(digest[0:8], "big")
    expected_lo = int.from_bytes(digest[8:16], "big")
    assert key_to_object_id(key) == (expected_hi, expected_lo)


def _error_with_status(status):
    exc = Exception("GET failed")
    exc.status = status  # what ddl_client.DdlError carries
    return exc


def test_is_not_found_error_matches_status_one():
    assert is_not_found_error(_error_with_status(1))


def test_is_not_found_error_rejects_other_statuses():
    assert not is_not_found_error(_error_with_status(-3))
    assert not is_not_found_error(_error_with_status(None))  # timeout / startup: no completion
    assert not is_not_found_error(Exception("some unrelated error"))
    # the message is no longer parsed
    assert not is_not_found_error(Exception("GET failed object=0x1:0x2 status=1 result_bytes=0"))
