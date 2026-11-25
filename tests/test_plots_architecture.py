"""Tests for the Chronos-2 architecture diagram of article 1.

The diagram describes the model, so the tests check two things: that it draws, and that every
size it prints comes from its ``Chronos2Sizes`` argument rather than from text frozen in the
code. The slow test loads the real checkpoint and asserts that the published defaults still
match it, which is what keeps the figure honest when a new checkpoint changes a limit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from c2p.plots.architecture import (
    DEFAULT_TITLE,
    FIGURE_WIDTH,
    FONT_PATH,
    Chronos2Sizes,
    Stage,
    chronos2_sizes,
    plot_architecture,
)


def figure_text(fig: Figure) -> str:
    """Every string drawn in the figure, joined, including the nested boxes and the badge."""
    return "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)


def test_default_figure_names_the_six_stages_and_the_published_sizes() -> None:
    fig = plot_architecture()
    text = figure_text(fig)
    try:
        for stage in (
            "Grouped inputs",
            "Robust scaling",
            "embedding",
            "T5-style encoder",
            "Quantile head",
            "Un-scaling",
        ):
            assert stage in text, stage
        for number in ("8,192", "1,024", "21 quantile", "768", "12 blocks", "64 output patches"):
            assert number in text, number
        assert "Group attention" in text and "Time attention" in text
        assert DEFAULT_TITLE in text
    finally:
        plt.close(fig)


def test_sizes_are_arguments_not_hard_coded_text() -> None:
    """A different checkpoint must change the figure: no size may be frozen in the code."""
    sizes = Chronos2Sizes(
        context_length=2048,
        prediction_length=64,
        n_quantiles=9,
        quantile_low=0.1,
        quantile_high=0.9,
        patch_size=8,
        n_blocks=6,
        d_model=384,
    )
    fig = plot_architecture(sizes=sizes, title=None)
    text = figure_text(fig)
    try:
        for number in ("2,048", "64", "9 quantile", "384", "6 blocks", "8 output patches"):
            assert number in text, number
        for stale in ("8,192", "1,024", "21 quantile", "768", "12 blocks"):
            assert stale not in text, stale
        assert DEFAULT_TITLE not in text
    finally:
        plt.close(fig)


def test_hand_drawn_style_uses_the_shipped_font_and_is_centred() -> None:
    """Excalifont ships with the package, every text uses it, and the column is centred."""
    assert FONT_PATH.is_file(), FONT_PATH
    assert (FONT_PATH.parent / "OFL.txt").is_file()
    fig = plot_architecture()
    try:
        (ax,) = fig.axes
        assert ax.texts, "no text drawn"
        for text in ax.texts:
            assert text.get_fontproperties().get_file() == str(FONT_PATH)
        title = next(text for text in ax.texts if text.get_text() == DEFAULT_TITLE)
        assert title.get_position()[0] == pytest.approx(FIGURE_WIDTH / 2)
        assert title.get_ha() == "center"
        boxes = [patch for patch in ax.patches if patch.get_sketch_params() is not None]
        assert boxes, "no hand-drawn outlines"
        widest = max(boxes, key=lambda patch: patch.get_window_extent().width)
        left, right = widest.get_window_extent().x0, widest.get_window_extent().x1
        canvas = ax.get_window_extent()
        assert left - canvas.x0 == pytest.approx(canvas.x1 - right, abs=2.0)
    finally:
        plt.close(fig)


def test_horizon_must_be_a_whole_number_of_patches() -> None:
    sizes = Chronos2Sizes(prediction_length=100, patch_size=16)
    with pytest.raises(ValueError, match="not a whole number of patch_size"):
        plot_architecture(sizes=sizes)


@pytest.mark.slow
def test_published_defaults_match_the_real_checkpoint() -> None:
    """``amazon/chronos-2`` still has the limits the default diagram prints."""
    from c2p.models import load_chronos2

    assert chronos2_sizes(load_chronos2(device="cpu")) == Chronos2Sizes()


# --- Generic stages and the decision path ---------------------------------------------------

STEPS = ("data", "baseline", "covariates", "context", "finetune", "calibration", "serving")


def test_plot_stages_draws_one_numbered_box_per_stage() -> None:
    from c2p.plots.architecture import plot_stages

    stages = [Stage("First", "what happens first"), Stage("Second", "and then", badge="new")]
    fig = plot_stages(stages, title="Two steps")
    texts = [text.get_text() for text in fig.axes[0].texts]
    assert "Two steps" in texts and "First" in texts and "Second" in texts and "new" in texts
    assert {"1", "2"} <= set(texts)
    with pytest.raises(ValueError, match="at least one stage"):
        plot_stages([])


def test_decision_path_prints_what_was_measured_for_every_step() -> None:
    from c2p.plots.architecture import decision_stages, plot_decision_path

    measured = {key: f"measured for {key}" for key in (*STEPS, "monitoring")}
    stages = decision_stages(measured)
    assert len(stages) == 8
    assert [stage.sub[0] for stage in stages] == [
        ("Measured", f"measured for {key}") for key in (*STEPS, "monitoring")
    ]
    fig = plot_decision_path(measured)
    drawn = " ".join(text.get_text() for text in fig.axes[0].texts)
    assert "measured for calibration" in drawn
    with pytest.raises(ValueError, match="monitoring"):
        decision_stages(dict.fromkeys(STEPS, "x"))
