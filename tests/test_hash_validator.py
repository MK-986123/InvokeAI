import json
from base64 import b64decode, b64encode

import pytest

from invokeai.backend.model_hash.hash_validator import hashes, validate_hash


def test_validate_hash_no_colon():
    # Hashes without colons should be ignored without raising exceptions
    validate_hash("invalidhash")
    validate_hash("")
    validate_hash("sha256:invalid:hash")


def test_validate_hash_allowed_hash():
    # Valid formatted hash that is not in the blocked list should pass silently
    validate_hash("sha256:0000000000000000000000000000000000000000000000000000000000000000")
    validate_hash("md5:1234567890abcdef1234567890abcdef")
    validate_hash("blake3:1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef")


def test_validate_hash_blocked_hash_actual():
    # Test using actual hardcoded blocked hashes in hash_validator.py
    assert len(hashes) > 0
    map0 = json.loads(b64decode(hashes[0]))

    # sha256 blocked hash
    sha256_hash = map0["sha256"]
    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash(f"sha256:{sha256_hash}")

    # blake3 alias for blake3_single
    blake3_hash = map0["blake3_single"]
    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash(f"blake3:{blake3_hash}")


def test_validate_hash_monkeypatched_hashes(monkeypatch: pytest.MonkeyPatch):
    # Test custom hashes using monkeypatch
    custom_map = {
        "sha256": "bad_sha256_hash",
        "blake3_single": "bad_blake3_hash",
    }
    encoded_custom_map = b64encode(json.dumps(custom_map).encode()).decode()

    monkeypatch.setattr("invokeai.backend.model_hash.hash_validator.hashes", [encoded_custom_map])

    # Unblocked hash should pass
    validate_hash("sha256:good_sha256_hash")

    # Blocked sha256 should raise Exception
    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash("sha256:bad_sha256_hash")

    # Blocked blake3 (aliased to blake3_single) should raise Exception
    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash("blake3:bad_blake3_hash")

    # Blocked blake3_single should raise Exception
    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash("blake3_single:bad_blake3_hash")
