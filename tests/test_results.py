from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from c2p.config import get_paths
from c2p.results import load_results, run_metadata, save_results


@pytest.fixture
def paths(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "chronos2-in-practice"\n')
    return get_paths(fast_mode=True, root=tmp_path)


def test_round_trip_parses_dates(paths) -> None:
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2018-01-01", periods=3, freq="h"), "value": [1.0, 2.0, 3.0]}
    )
    path = save_results(df, "00_example", paths)
    assert path == paths.results / "00_example.csv"
    loaded = load_results("00_example", paths)
    pd.testing.assert_frame_equal(loaded, df, check_dtype=False)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])


@pytest.mark.parametrize("name", ["example", "0_x", "00-Example", "00_bad name", "../00_x"])
def test_rejects_bad_names(paths, name: str) -> None:
    with pytest.raises(ValueError, match="must look like"):
        save_results(pd.DataFrame({"a": [1]}), name, paths)


def test_load_missing_file_explains(paths) -> None:
    with pytest.raises(FileNotFoundError, match="run the cell"):
        load_results("01_missing", paths)


def test_run_metadata_lists_versions() -> None:
    meta = run_metadata(device="cpu", fast_mode=True)
    assert meta["chronos-forecasting"] == "2.3.2"
    assert meta["device"] == "cpu"
    assert meta["fast_mode"] == "True"
