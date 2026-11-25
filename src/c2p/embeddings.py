"""Chronos-2 as an encoder: embeddings of time-series windows and four things to do with them.

``Chronos2Pipeline.embed`` returns what the encoder computed for a series before any forecast
is decoded: one vector of ``d_model`` numbers per 16-step patch of the context, plus two
special tokens. This module turns that into one vector per window and adds the small, plain
tools needed to ask whether the vector is useful: nearest neighbours, a linear probe, an
anomaly score, and the ROC AUC to judge it. Everything after :func:`embed_windows` is NumPy
on ordinary arrays, so every function also runs on the raw windows, which is the control each
experiment needs: an embedding earns its place only if it beats the numbers it was made from.

Two properties of the embedding to keep in mind:

- **The level is gone.** Chronos-2 standardises each series before encoding it (``embed``
  returns the location and scale separately), so two windows with the same shape at different
  levels have nearly the same embedding. Use :func:`standardize_windows` on raw windows for a
  like-for-like control.
- **Padding is embedded too.** Windows of different lengths are left-padded to the longest,
  and the padded patches get vectors. :func:`embed_windows` therefore takes one array of
  equal-length windows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.stats

if TYPE_CHECKING:
    from chronos import Chronos2Pipeline

__all__ = [
    "PATCH",
    "POOLS",
    "analog_forecast",
    "cosine_neighbors",
    "embed_windows",
    "knn_score",
    "ridge_probe",
    "roc_auc",
    "sliding_windows",
    "standardize_windows",
]

Pool = Literal["mean", "reg", "forecast"]
POOLS: tuple[Pool, ...] = ("mean", "reg", "forecast")
PATCH = 16  # steps per input patch of Chronos-2: use window lengths that are multiples of it


def sliding_windows(values: np.ndarray, *, length: int, ends: Sequence[int]) -> np.ndarray:
    """Stack the windows ``values[end - length:end]`` for every ``end`` (exclusive index)."""
    values = np.asarray(values, dtype="float64")
    ends = [int(end) for end in ends]
    if length < 1:
        raise ValueError(f"length must be positive, got {length}.")
    if not ends:
        raise ValueError("ends is empty.")
    if min(ends) < length or max(ends) > len(values):
        raise ValueError(
            f"Every end must lie in [{length}, {len(values)}]; got {min(ends)} to {max(ends)}."
        )
    return np.stack([values[end - length : end] for end in ends])


def standardize_windows(windows: np.ndarray) -> np.ndarray:
    """Each window minus its mean, divided by its standard deviation (NaN-aware).

    This is the control for the embedding, which is computed on a standardised series: the
    same information the encoder starts from, without the encoder.
    """
    windows = np.asarray(windows, dtype="float64")
    mean = np.nanmean(windows, axis=1, keepdims=True)
    scale = np.nanstd(windows, axis=1, keepdims=True)
    return (windows - mean) / np.where(scale > 0, scale, 1.0)


def embed_windows(
    pipeline: Chronos2Pipeline,
    windows: np.ndarray,
    *,
    pool: Pool = "mean",
    batch_size: int = 256,
) -> np.ndarray:
    """One embedding per window: an array of shape ``(n_windows, d_model)``.

    Parameters
    ----------
    pipeline
        A loaded ``Chronos2Pipeline``.
    windows
        Array of shape ``(n_windows, length)``, one univariate window per row. NaN marks a
        missing value, as in ``predict``.
    pool
        How the encoder's tokens become one vector. ``"mean"`` averages the vectors of the
        context patches; ``"reg"`` takes the [REG] token that closes the context;
        ``"forecast"`` takes the token of the first future patch, the one the forecast head
        reads.
    batch_size
        Windows per forward pass.
    """
    import torch

    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}.")
    windows = np.asarray(windows, dtype="float32")
    if windows.ndim != 2:
        raise ValueError(f"windows must have shape (n_windows, length), got {windows.shape}.")
    embeddings, _ = pipeline.embed(torch.from_numpy(windows[:, None, :]), batch_size=batch_size)
    tokens = torch.stack([embedding[0] for embedding in embeddings]).float()
    if pool == "mean":
        pooled = tokens[:, :-2].mean(dim=1)
    elif pool == "reg":
        pooled = tokens[:, -2]
    else:
        pooled = tokens[:, -1]
    return pooled.cpu().numpy().astype("float64")


def _unit(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype="float64")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.where(norms > 0, norms, 1.0)


def cosine_neighbors(
    queries: np.ndarray, library: np.ndarray, *, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` rows of ``library`` most similar to each query, most similar first.

    Returns ``(indices, similarities)``, both of shape ``(n_queries, k)``; the similarity is
    the cosine, 1 for the same direction.
    """
    if not 1 <= k <= len(library):
        raise ValueError(f"k must be between 1 and the library size {len(library)}, got {k}.")
    similarity = _unit(queries) @ _unit(library).T
    indices = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    return indices, np.take_along_axis(similarity, indices, axis=1)


def knn_score(queries: np.ndarray, library: np.ndarray, *, k: int = 10) -> np.ndarray:
    """Anomaly score of each query: mean cosine distance to its ``k`` nearest library rows.

    A window that resembles nothing in the library scores high. The library should hold
    ordinary windows only, and must not contain the queries themselves.
    """
    _, similarity = cosine_neighbors(queries, library, k=k)
    return 1.0 - similarity.mean(axis=1)


def analog_forecast(
    queries: np.ndarray,
    library: np.ndarray,
    *,
    query_windows: np.ndarray,
    library_windows: np.ndarray,
    library_futures: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Forecast each query's future as the average future of its ``k`` most similar windows.

    ``queries`` and ``library`` are the representations the similarity is computed on
    (embeddings, or standardised raw windows). The futures of the analogs are expressed in
    units of their own window (minus its mean, over its standard deviation), averaged, and
    brought back to the level and scale of the query's window, so an analog from a
    low-demand year can stand in for a high-demand one.

    Returns an array of shape ``(n_queries, horizon)``.
    """
    indices, _ = cosine_neighbors(queries, library, k=k)
    library_windows = np.asarray(library_windows, dtype="float64")
    library_futures = np.asarray(library_futures, dtype="float64")
    query_windows = np.asarray(query_windows, dtype="float64")
    mean = np.nanmean(library_windows, axis=1, keepdims=True)
    scale = np.nanstd(library_windows, axis=1, keepdims=True)
    relative = (library_futures - mean) / np.where(scale > 0, scale, 1.0)
    shape = np.nanmean(relative[indices], axis=1)
    level = np.nanmean(query_windows, axis=1, keepdims=True)
    spread = np.nanstd(query_windows, axis=1, keepdims=True)
    return level + spread * shape


def ridge_probe(
    train_x: np.ndarray,
    train_y: Sequence[object],
    test_x: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Predict a label for each test row with a linear (ridge) classifier fitted on the train rows.

    A linear probe asks whether a property can be read off a representation with nothing
    more than a weighted sum: features are standardised with the training statistics, one
    ridge regression per class is fitted against a 0/1 indicator, and the class with the
    largest output wins. ``alpha`` is the ridge penalty relative to the number of training
    rows. Returns the predicted labels, in the dtype of ``train_y``.
    """
    train_x = np.asarray(train_x, dtype="float64")
    test_x = np.asarray(test_x, dtype="float64")
    labels = np.asarray(train_y)
    if len(labels) != len(train_x):
        raise ValueError(f"{len(train_x)} training rows but {len(labels)} labels.")
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    classes, codes = np.unique(labels, return_inverse=True)
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    x = (train_x - mean) / scale
    targets = np.eye(len(classes))[codes]
    offset = targets.mean(axis=0)
    # Dual form: with more features than rows, solve the (rows x rows) system.
    gram = x @ x.T + alpha * len(x) * np.eye(len(x))
    weights = x.T @ np.linalg.solve(gram, targets - offset)
    scores = ((test_x - mean) / scale) @ weights + offset
    return classes[np.argmax(scores, axis=1)]


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve: the chance that a positive scores above a negative (ties 0.5)."""
    positive = np.asarray(labels, dtype=bool)
    n_positive, n_negative = int(positive.sum()), int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        raise ValueError("roc_auc needs at least one positive and one negative label.")
    ranks = scipy.stats.rankdata(np.asarray(scores, dtype="float64"))
    return float(
        (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    )
