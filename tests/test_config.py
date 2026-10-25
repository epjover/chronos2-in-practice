from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from c2p import config


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "{config.PROJECT_NAME}"\n')
    (tmp_path / "notebooks").mkdir()
    return tmp_path


def test_find_repo_root_walks_up_from_subfolder(fake_repo: Path, monkeypatch) -> None:
    monkeypatch.delenv("C2P_ROOT", raising=False)
    assert config.find_repo_root(fake_repo / "notebooks") == fake_repo.resolve()


def test_find_repo_root_ignores_other_projects(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("C2P_ROOT", raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "something-else"\n')
    with pytest.raises(FileNotFoundError, match="C2P_ROOT"):
        config.find_repo_root(tmp_path)


def test_find_repo_root_env_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("C2P_ROOT", str(tmp_path))
    assert config.find_repo_root() == tmp_path.resolve()


def test_get_paths_separates_fast_mode_results(fake_repo: Path) -> None:
    full = config.get_paths(fast_mode=False, root=fake_repo)
    fast = config.get_paths(fast_mode=True, root=fake_repo)
    assert full.results == fake_repo / "results"
    assert fast.results == fake_repo / "results" / "fast"
    assert fast.figures == fast.results / "figures"
    for folder in (full.data_raw, full.results, full.figures, full.models, fast.figures):
        assert folder.is_dir()


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [(None, False, False), (None, True, True), ("1", False, True), ("false", True, False)],
)
def test_is_fast_mode(monkeypatch, value: str | None, default: bool, expected: bool) -> None:
    if value is None:
        monkeypatch.delenv("C2P_FAST_MODE", raising=False)
    else:
        monkeypatch.setenv("C2P_FAST_MODE", value)
    assert config.is_fast_mode(default=default) is expected


def test_is_fast_mode_rejects_garbage(monkeypatch) -> None:
    monkeypatch.setenv("C2P_FAST_MODE", "maybe")
    with pytest.raises(ValueError, match="C2P_FAST_MODE"):
        config.is_fast_mode()


def test_set_seed_makes_global_numpy_rng_reproducible() -> None:
    config.set_seed(7)
    first = np.random.rand(3)  # noqa: NPY002 - the global RNG is what set_seed must control
    config.set_seed(7)
    np.testing.assert_array_equal(first, np.random.rand(3))  # noqa: NPY002


def test_get_device_env_override(monkeypatch) -> None:
    monkeypatch.setenv("C2P_DEVICE", "cpu")
    assert config.get_device() == "cpu"
    monkeypatch.setenv("C2P_DEVICE", "tpu")
    with pytest.raises(ValueError, match="C2P_DEVICE"):
        config.get_device()


def test_get_device_without_override(monkeypatch) -> None:
    monkeypatch.delenv("C2P_DEVICE", raising=False)
    assert config.get_device(prefer_gpu=False) == "cpu"
    assert config.get_device() in {"cpu", "cuda"}
