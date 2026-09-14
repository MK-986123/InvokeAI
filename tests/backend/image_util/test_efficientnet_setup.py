import re
from pathlib import Path

SETUP_PY = Path("invokeai/backend/image_util/normal_bae/nets/submodules/efficientnet_repo/setup.py")
VERSION_PY = Path("invokeai/backend/image_util/normal_bae/nets/submodules/efficientnet_repo/geffnet/version.py")


def test_setup_py_no_exec():
    content = SETUP_PY.read_text(encoding="utf-8")
    assert "exec(" not in content, "setup.py should not contain exec()"


def test_version_extraction_regex():
    version_content = VERSION_PY.read_text(encoding="utf-8")
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", version_content, re.M)
    assert version_match is not None
    assert version_match.group(1) == "1.0.2"


def test_version_extraction_safe_against_code_execution():
    malicious_version_file_content = "__version__ = '1.0.2'\nraise RuntimeError('Malicious code executed!')"
    version_match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", malicious_version_file_content, re.M)
    assert version_match is not None
    assert version_match.group(1) == "1.0.2"
