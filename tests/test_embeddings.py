"""Tests for ``c2p.embeddings``: pooling, neighbours, analogs, the linear probe and the AUC."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from c2p.embeddings import (
    POOLS,
    analog_forecast,
    cosine_neighbors,
    embed_windows,
    knn_score,
    ridge_probe,
    roc_auc,
    sliding_windows,
    standardize_windows,
)


class StubPipeline:
    """``embed`` returns tokens whose values name their position: patches 0.., then 100, 200."""

    d_model = 3

    def embed(self, inputs, batch_size=256):
        n_windows, _, length = inputs.shape
        n_patches = length // 16
        tokens = torch.arange(n_patches, dtype=torch.float32)
        tokens = torch.cat([tokens, torch.tensor([100.0, 200.0])])
        one = tokens[None, :, None].expand(1, -1, self.d_model)
        embeddings = [one + index for index in range(n_windows)]
        return embeddings, [(torch.zeros(1), torch.ones(1))] * n_windows


def test_sliding_windows_end_at_the_given_indices() -> None:
    windows = sliding_windows(np.arange(10.0), length=3, ends=[3, 10])
    assert windows.tolist() == [[0.0, 1.0, 2.0], [7.0, 8.0, 9.0]]
    with pytest.raises(ValueError, match="Every end"):
        sliding_windows(np.arange(10.0), length=3, ends=[2])
    with pytest.raises(ValueError, match="empty"):
        sliding_windows(np.arange(10.0), length=3, ends=[])


def test_standardize_windows_removes_level_and_scale() -> None:
    windows = np.array([[1.0, 2.0, 3.0], [10.0, 30.0, np.nan], [5.0, 5.0, 5.0]])
    out = standardize_windows(windows)
    assert out[0] == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert out[1, :2] == pytest.approx([-1.0, 1.0]) and np.isnan(out[1, 2])
    assert out[2].tolist() == [0.0, 0.0, 0.0]  # a constant window: no division by zero


@pytest.mark.parametrize(("pool", "first"), [("mean", 1.5), ("reg", 100.0), ("forecast", 200.0)])
def test_embed_windows_pools_the_right_tokens(pool: str, first: float) -> None:
    windows = np.zeros((2, 64))  # four patches: tokens 0, 1, 2, 3, then [REG] and the future
    out = embed_windows(StubPipeline(), windows, pool=pool)
    assert out.shape == (2, 3)
    assert out[0].tolist() == [first] * 3
    assert out[1].tolist() == [first + 1] * 3  # the second window, in order


def test_embed_windows_validates_its_inputs() -> None:
    assert POOLS == ("mean", "reg", "forecast")
    with pytest.raises(ValueError, match="pool"):
        embed_windows(StubPipeline(), np.zeros((2, 64)), pool="max")
    with pytest.raises(ValueError, match="shape"):
        embed_windows(StubPipeline(), np.zeros(64))


def test_cosine_neighbors_rank_by_direction_not_length() -> None:
    library = np.array([[1.0, 0.0], [0.0, 5.0], [10.0, 10.0]])
    indices, similarity = cosine_neighbors(np.array([[3.0, 0.1]]), library, k=2)
    assert indices.tolist() == [[0, 2]]
    assert similarity[0, 0] > similarity[0, 1] > 0.7
    with pytest.raises(ValueError, match="k must be"):
        cosine_neighbors(library, library, k=4)


def test_knn_score_is_high_for_a_window_unlike_the_library() -> None:
    rng = np.random.default_rng(0)
    library = np.array([1.0, 1.0, 0.0]) + 0.05 * rng.normal(size=(50, 3))
    usual = np.array([[1.0, 1.0, 0.0]])
    odd = np.array([[0.0, 0.0, 1.0]])
    assert knn_score(odd, library, k=5)[0] > 0.8 > 0.01 > knn_score(usual, library, k=5)[0]


def test_analog_forecast_rescales_the_future_of_the_nearest_windows() -> None:
    library_windows = np.array([[0.0, 2.0], [0.0, 2.0], [5.0, 1.0]])
    library_futures = np.array([[4.0], [4.0], [9.0]])  # (4 - 1) / 1 = 3 window-sds above the mean
    library = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0]])
    query_windows = np.array([[10.0, 30.0]])  # mean 20, sd 10
    forecast = analog_forecast(
        np.array([[1.0, 0.05]]),
        library,
        query_windows=query_windows,
        library_windows=library_windows,
        library_futures=library_futures,
        k=2,
    )
    assert forecast.tolist() == [[50.0]]  # 20 + 10 * 3


def test_ridge_probe_reads_a_linear_label_and_keeps_the_label_dtype() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 20))
    labels = np.where(x[:, 0] + 0.5 * x[:, 1] > 0, "up", "down")
    predicted = ridge_probe(x[:200], labels[:200], x[200:], alpha=0.1)
    assert (predicted == labels[200:]).mean() > 0.9
    assert set(predicted) <= {"up", "down"}
    # More features than rows uses the same code path.
    wide = rng.normal(size=(30, 100))
    assert ridge_probe(wide, np.arange(30) % 3, wide).shape == (30,)
    with pytest.raises(ValueError, match="labels"):
        ridge_probe(x[:10], labels[:5], x)
    with pytest.raises(ValueError, match="alpha"):
        ridge_probe(x[:10], labels[:10], x, alpha=0.0)


def test_roc_auc_is_the_probability_of_ranking_a_positive_above_a_negative() -> None:
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert roc_auc([1, 2, 3, 4], [False, True, False, True]) == 0.75
    with pytest.raises(ValueError, match="at least one"):
        roc_auc([0.1, 0.2], [True, True])
