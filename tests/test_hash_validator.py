import json
from base64 import b64decode

import pytest

from invokeai.backend.model_hash.hash_validator import hashes, validate_hash


def test_validate_hash_happy_path():
    # Should not raise an exception
    validate_hash("sha256:some_valid_hash")
    validate_hash("md5:abc123xyz")
    validate_hash("blake3_single:fake_hash")


def test_validate_hash_invalid_format():
    # Should return early without raising an exception
    validate_hash("invalid_format_without_colon")


def test_validate_hash_blacklist_trigger():
    # Pick the first hash from the blacklist
    enc_hash = hashes[0]
    hash_map = json.loads(b64decode(enc_hash))

    # Take an algorithm and its hash from the map (e.g., sha256)
    alg = "sha256"
    bad_hash = hash_map[alg]

    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash(f"{alg}:{bad_hash}")


def test_validate_hash_blake3_override():
    # Ensure that `blake3` is overridden to `blake3_single` and triggers the blacklist
    enc_hash = hashes[0]
    hash_map = json.loads(b64decode(enc_hash))

    # The map uses 'blake3_single'
    bad_hash = hash_map["blake3_single"]

    # We pass 'blake3', but it should be mapped to 'blake3_single' internally and raise
    with pytest.raises(Exception, match="This model can not be loaded"):
        validate_hash(f"blake3:{bad_hash}")
