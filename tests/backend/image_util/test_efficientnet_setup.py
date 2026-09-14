import runpy
from pathlib import Path
from unittest.mock import Mock

import setuptools

REPO_ROOT = Path(__file__).parents[3]
SETUP_PY = REPO_ROOT / "invokeai/backend/image_util/normal_bae/nets/submodules/efficientnet_repo/setup.py"


def _run_setup_py(monkeypatch, setup_py: Path) -> dict:
    setup_mock = Mock()
    monkeypatch.setattr(setuptools, "setup", setup_mock)
    monkeypatch.setattr(setuptools, "find_packages", lambda exclude=None: [])
    runpy.run_path(str(setup_py), run_name="__main__")
    return setup_mock.call_args.kwargs


def test_setup_py_uses_version_from_version_file(monkeypatch):
    setup_kwargs = _run_setup_py(monkeypatch, SETUP_PY)

    assert setup_kwargs["name"] == "geffnet"
    assert setup_kwargs["version"] == "1.0.2"


def test_setup_py_does_not_execute_version_file_code(monkeypatch, tmp_path):
    temp_repo = tmp_path / "efficientnet_repo"
    temp_repo.mkdir()
    (temp_repo / "README.md").write_text("Temporary README", encoding="utf-8")

    geffnet_dir = temp_repo / "geffnet"
    geffnet_dir.mkdir()
    (geffnet_dir / "version.py").write_text(
        "__version__ = '1.0.2'\nraise RuntimeError('Malicious code executed!')\n",
        encoding="utf-8",
    )

    temp_setup_py = temp_repo / "setup.py"
    temp_setup_py.write_text(SETUP_PY.read_text(encoding="utf-8"), encoding="utf-8")

    setup_kwargs = _run_setup_py(monkeypatch, temp_setup_py)

    assert setup_kwargs["version"] == "1.0.2"
