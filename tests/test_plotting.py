"""Smoke and contract tests for ``c2p.plotting`` on synthetic frames (Agg backend, no network)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.text import Text
from pandas.tseries.frequencies import to_offset

from c2p import plotting
from c2p.schema import (
    EVAL_QUANTILES,
    ID_COL,
    PREDICTION_COL,
    TARGET_COL,
    TARGET_NAME_COL,
    TIMESTAMP_COL,
    quantile_col,
)

BENCHMARKS_CSV = Path(__file__).resolve().parents[1] / "data" / "reference" / "paper_benchmarks.csv"
HORIZON = 24


@pytest.fixture(autouse=True)
def _agg_backend() -> Iterator[None]:
    plt.switch_backend("Agg")
    yield
    plt.close("all")


# --- Synthetic data -----------------------------------------------------------------------


def hourly_frame(n: int, *, item: str = "ES", start: str = "2018-03-01") -> pd.DataFrame:
    """Load-like hourly series: ~25k level, daily cycle, lower weekends."""
    ts = pd.date_range(start, periods=n, freq="h")
    hour = ts.hour.to_numpy()
    weekend = ts.day_of_week.to_numpy() >= 5
    values = 25_000 + 3_000 * np.sin(2 * np.pi * hour / 24) - 2_500 * weekend
    return pd.DataFrame({ID_COL: item, TIMESTAMP_COL: ts, TARGET_COL: values.astype(float)})


def frame_on(timestamps: pd.DatetimeIndex, *, item: str = "ES") -> pd.DataFrame:
    """Smooth series on any grid (e.g. business days or month starts)."""
    values = 25_000 + 3_000 * np.sin(np.arange(len(timestamps)) / 4)
    return pd.DataFrame({ID_COL: item, TIMESTAMP_COL: timestamps, TARGET_COL: values})


def forecast_for(context: pd.DataFrame, *, horizon: int = HORIZON, freq: str = "h") -> pd.DataFrame:
    """Forecast frame after the end of each series, with every evaluation quantile."""
    blocks = []
    for item, rows in context.groupby(ID_COL, sort=False):
        last = rows[TIMESTAMP_COL].max()
        median = np.full(horizon, 25_000.0)
        block = pd.DataFrame(
            {
                ID_COL: item,
                TIMESTAMP_COL: pd.date_range(last, periods=horizon + 1, freq=freq)[1:],
                TARGET_NAME_COL: TARGET_COL,
                PREDICTION_COL: median,
            }
        )
        for q in EVAL_QUANTILES:
            block[quantile_col(q)] = median + (q - 0.5) * 4_000 * np.linspace(1, 2, horizon)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


@pytest.fixture
def series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(context, forecast, actuals) for one hourly series."""
    full = hourly_frame(24 * 21)
    context = full.iloc[:-HORIZON].reset_index(drop=True)
    actuals = full.iloc[-HORIZON:].reset_index(drop=True)
    return context, forecast_for(context), actuals


# --- Assertions ---------------------------------------------------------------------------


def legend_labels(legend) -> list[str]:
    assert legend is not None, "expected a legend"
    return [text.get_text() for text in legend.get_texts()]


def assert_no_dual_axes(fig: Figure) -> None:
    """Twin axes share the position of their host; every visible axes must be distinct."""
    positions = [
        tuple(np.round(ax.get_position().bounds, 6)) for ax in fig.axes if ax.get_visible()
    ]
    assert len(positions) == len(set(positions)), "found overlapping (twin) axes"


def assert_saved(fig: Figure, folder: Path, name: str) -> Path:
    (path,) = plotting.save_figure(fig, name, folder)
    assert path == folder / f"{name}.png"
    assert path.is_file()
    assert path.stat().st_size > 1_000
    return path


def plot_area_pixels(fig: Figure, ax: Axes, *, dpi: int = 200) -> np.ndarray:
    """RGB pixels inside ``ax`` (2 px trimmed on each side) when rendered at ``dpi``."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)
    box = ax.get_window_extent()
    height = image.shape[0]
    rows = slice(height - int(box.y1) + 2, height - int(box.y0) - 2)
    return image[rows, int(box.x0) + 2 : int(box.x1) - 2]


def pixel_share(pixels: np.ndarray, color: str, *, tolerance: int = 30) -> float:
    """Share of ``pixels`` within ``tolerance`` (sum of RGB differences) of ``color``."""
    target = np.array(to_rgb(color)) * 255
    return float((np.abs(pixels - target).sum(axis=2) < tolerance).mean())


def reading_order(texts: list[Text]) -> list[str]:
    """Texts sorted top to bottom, then left to right, as rendered."""
    boxes = [(text.get_window_extent(), text.get_text()) for text in texts]
    return [label for box, label in sorted(boxes, key=lambda b: (-round(b[0].y0), b[0].x0))]


# --- Style --------------------------------------------------------------------------------


def test_set_style_is_idempotent() -> None:
    with mpl.rc_context():
        plotting.set_style()
        first = {key: mpl.rcParams[key] for key in plotting.STYLE}
        plotting.set_style()
        second = {key: mpl.rcParams[key] for key in plotting.STYLE}
    assert first == second
    assert first["font.sans-serif"][:2] == ["Segoe UI", "DejaVu Sans"]


def test_style_context_restores_rcparams() -> None:
    with mpl.rc_context({"axes.titlelocation": "center", "lines.linewidth": 3.0}):
        with plotting.style_context():
            assert mpl.rcParams["axes.titlelocation"] == "left"
            assert mpl.rcParams["lines.linewidth"] == plotting.LINE_WIDTH
        assert mpl.rcParams["axes.titlelocation"] == "center"
        assert mpl.rcParams["lines.linewidth"] == 3.0


def test_missing_first_font_falls_back_without_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="matplotlib.font_manager")
    with (
        plotting.style_context(),
        mpl.rc_context({"font.sans-serif": ["A Font That Does Not Exist", "DejaVu Sans"]}),
    ):
        fig, ax = plt.subplots()
        ax.set_title("Fallback font")
        fig.canvas.draw()
    assert not [record for record in caplog.records if "findfont" in record.getMessage()]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (25_000, "25K"),
        (22_500, "22.5K"),
        (25_050, "25.05K"),
        (1_500_000, "1.5M"),
        (0.05, "0.05"),
        (-3_000, "-3K"),
        (999, "999"),
        (999_995, "1M"),
        (999_999, "1M"),
        (-999_999, "-1M"),
    ],
)
def test_format_compact(value: float, expected: str) -> None:
    assert plotting.format_compact(value) == expected


@pytest.mark.parametrize(
    ("ticks", "expected"),
    [
        ([24_000, 25_000, 26_000], ["24K", "25K", "26K"]),
        ([25_000, 25_002, 25_004], ["25,000", "25,002", "25,004"]),
        ([1_000.0, 1_000.5, 1_001.0], ["1,000.0", "1,000.5", "1,001.0"]),
    ],
)
def test_compact_formatter_keeps_ticks_distinct(ticks: list[float], expected: list[str]) -> None:
    assert plotting._CompactFormatter().format_ticks(ticks) == expected


# --- Colours ------------------------------------------------------------------------------


def test_assign_colors_keeps_model_colors_and_fills_free_slots() -> None:
    colors = plotting.assign_colors(["my-model", "chronos-2", "seasonal_naive"])
    assert colors["chronos-2"] == plotting.MODEL_COLORS["chronos-2"]
    assert colors["seasonal_naive"] == plotting.MODEL_COLORS["seasonal_naive"]
    assert colors["my-model"] not in plotting.MODEL_COLORS.values()
    assert colors["my-model"] == plotting.PALETTE[len(plotting.MODEL_COLORS)]


def test_assign_colors_never_lends_a_model_slot_to_a_new_model() -> None:
    alone = plotting.assign_colors(["chronos-2-lora", "chronos-2-full"])
    beside = plotting.assign_colors(["chronos-2", "chronos-2-lora", "chronos-2-full"])
    assert not set(alone.values()) & set(plotting.MODEL_COLORS.values())
    assert alone == {name: beside[name] for name in alone}, "same colour in every figure"


def test_assign_colors_without_fixed_entities_starts_at_slot_one() -> None:
    assert plotting.assign_colors(["a", "b"], fixed={}) == {
        "a": plotting.PALETTE[0],
        "b": plotting.PALETTE[1],
    }


def test_assign_colors_never_cycles() -> None:
    with pytest.raises(ValueError, match="palette slots"):
        plotting.assign_colors([f"series-{i}" for i in range(len(plotting.PALETTE) + 1)])


# --- save_figure --------------------------------------------------------------------------


def test_save_figure_writes_every_format_and_closes(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    paths = plotting.save_figure(fig, "demo", tmp_path / "nested", formats=("png", "svg"))
    assert [path.name for path in paths] == ["demo.png", "demo.svg"]
    assert all(path.stat().st_size > 0 for path in paths)
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "sub/demo",
        "sub\\demo",
        " demo",
        "fan chart",
        "demo.",
        "wql:ctx",  # an NTFS alternate data stream on Windows: the PNG would be hidden
        "fan.png",  # would be written as fan.png.png
        "CON",
        "con.png",
    ],
)
def test_save_figure_rejects_bad_names_and_keeps_figure_open(tmp_path: Path, name: str) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="Figure name"):
        plotting.save_figure(fig, name, tmp_path)
    assert plt.fignum_exists(fig.number)


def test_save_figure_accepts_dotted_stems(tmp_path: Path) -> None:
    fig, _ = plt.subplots()
    (path,) = plotting.save_figure(fig, "fan_chart-v1.2", tmp_path)
    assert path.name == "fan_chart-v1.2.png"
    assert path.is_file()


@pytest.mark.parametrize("formats", [("bmpx",), "png", ()])
def test_save_figure_rejects_bad_formats(tmp_path: Path, formats) -> None:
    fig, _ = plt.subplots()
    with pytest.raises(ValueError, match="format"):
        plotting.save_figure(fig, "demo", tmp_path, formats=formats)


# --- plot_fan -----------------------------------------------------------------------------


def test_plot_fan_smoke(series, tmp_path: Path) -> None:
    context, forecast, actuals = series
    ax = plotting.plot_fan(context, forecast, actuals=actuals, title="Fan chart", ylabel="MW")
    fig = ax.figure
    assert legend_labels(ax.get_legend()) == [
        "History",
        "Actual",
        "Forecast (median)",
        "80% interval",
        "98% interval",
    ]
    assert ax.get_title(loc="left") == "Fan chart"
    assert ax.get_ylabel() == "MW"
    assert len(fig.axes) == 1
    assert_no_dual_axes(fig)
    median_handle = ax.get_legend().legend_handles[2]
    assert to_hex(median_handle.get_color()) == plotting.BLUE
    assert_saved(fig, tmp_path, "fan")


def test_plot_fan_history_limits_the_x_range(series) -> None:
    context, forecast, _ = series
    ax = plotting.plot_fan(context, forecast, history=48)
    expected_start = context[TIMESTAMP_COL].max() - pd.Timedelta(hours=47)
    expected_end = forecast[TIMESTAMP_COL].max()
    start, end = ax.get_xlim()
    assert start == pytest.approx(mdates.date2num(expected_start))
    assert end == pytest.approx(mdates.date2num(expected_end))


def test_plot_fan_draws_into_given_axes_without_intervals(series) -> None:
    context, forecast, _ = series
    fig, ax = plt.subplots()
    returned = plotting.plot_fan(context, forecast, intervals={}, ax=ax)
    assert returned is ax
    assert legend_labels(ax.get_legend()) == ["History", "Forecast (median)"]
    assert len(fig.axes) == 1


def test_plot_fan_needs_item_id_for_several_series() -> None:
    context = pd.concat([hourly_frame(200, item="A"), hourly_frame(200, item="B")])
    forecast = forecast_for(context)
    with pytest.raises(ValueError, match="item_id"):
        plotting.plot_fan(context, forecast)
    ax = plotting.plot_fan(context, forecast, item_id="B")
    assert ax.get_legend() is not None
    with pytest.raises(ValueError, match="not found"):
        plotting.plot_fan(context, forecast, item_id="C")


def test_plot_fan_rejects_missing_quantile_columns(series) -> None:
    context, forecast, _ = series
    with pytest.raises(ValueError, match=r"'0\.01'"):
        plotting.plot_fan(context, forecast.drop(columns=quantile_col(0.01)))


def test_plot_fan_rejects_forecast_overlapping_context(series) -> None:
    context, forecast, _ = series
    shifted = forecast.assign(**{TIMESTAMP_COL: forecast[TIMESTAMP_COL] - pd.Timedelta(hours=5)})
    with pytest.raises(ValueError, match="not after the context end"):
        plotting.plot_fan(context, shifted)


def test_plot_fan_rejects_unknown_target(series) -> None:
    context, forecast, _ = series
    renamed = context.rename(columns={TARGET_COL: "load"})
    with pytest.raises(ValueError, match="no rows for target 'load'"):
        plotting.plot_fan(renamed, forecast, target="load")


def test_plot_fan_rejects_timezone_aware_timestamps(series) -> None:
    context, forecast, _ = series
    aware = context.assign(**{TIMESTAMP_COL: context[TIMESTAMP_COL].dt.tz_localize("UTC")})
    with pytest.raises(TypeError, match="timezone-naive"):
        plotting.plot_fan(aware, forecast)


def test_plot_fan_rejects_more_than_three_intervals(series) -> None:
    context, forecast, _ = series
    intervals = {0.2: (0.4, 0.6), 0.4: (0.3, 0.7), 0.8: (0.1, 0.9), 0.98: (0.01, 0.99)}
    with pytest.raises(ValueError, match="nested intervals"):
        plotting.plot_fan(context, forecast, intervals=intervals)


def test_plot_fan_infers_a_single_non_string_item_id(series) -> None:
    context, forecast, _ = series
    ax = plotting.plot_fan(context.assign(**{ID_COL: 0}), forecast.assign(**{ID_COL: 0}))
    assert ax.get_legend() is not None


@pytest.mark.parametrize("freq", ["B", "MS"])
def test_plot_fan_counts_history_in_calendar_steps(freq: str) -> None:
    timestamps = pd.date_range("2010-01-01", periods=120, freq=freq)
    context = frame_on(timestamps)
    ax = plotting.plot_fan(context, forecast_for(context, horizon=6, freq=freq), history=48)
    start, _ = ax.get_xlim()
    assert start == pytest.approx(mdates.date2num(timestamps[-48]))


def test_plot_fan_asks_for_freq_when_a_daily_context_misses_rows() -> None:
    context = frame_on(pd.date_range("2018-01-01", periods=60, freq="D"))
    forecast = forecast_for(context, freq="D")
    gappy = context.drop(index=[30, 31]).reset_index(drop=True)
    with pytest.raises(ValueError, match="pass freq="):
        plotting.plot_fan(gappy, forecast)
    assert plotting.plot_fan(gappy, forecast, freq="D").get_legend() is not None


def test_plot_fan_rejects_a_context_off_the_given_freq(series) -> None:
    context, forecast, _ = series
    with pytest.raises(ValueError, match="off the '2h' grid"):
        plotting.plot_fan(context, forecast, freq="2h")


def test_plot_fan_keeps_narrow_y_ticks_distinct(series) -> None:
    context, forecast, _ = series
    narrow = context.assign(**{TARGET_COL: 25_000 + np.arange(len(context)) % 9})
    quantiles = [quantile_col(q) for q in EVAL_QUANTILES]
    tight = forecast.assign(**{PREDICTION_COL: 25_004.0}, **dict.fromkeys(quantiles, 25_004.0))
    ax = plotting.plot_fan(narrow, tight, history=72)
    ax.figure.canvas.draw()
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert len(labels) >= 3
    assert len(set(labels)) == len(labels), labels


# --- plot_scenarios -----------------------------------------------------------------------


@pytest.fixture
def scenarios(series):
    context, forecast, actuals = series
    n = len(context)
    positions = np.r_[n - 100 : n - 94, n - 40 : n - 36]  # two gaps: 6 and 4 hours
    as_nan = context.copy()
    as_nan.loc[positions, TARGET_COL] = np.nan
    dropped = context.drop(index=positions).reset_index(drop=True)
    spiked = context.copy()
    spiked.loc[n - 20, TARGET_COL] *= 1.5
    spike_at = context.loc[n - 20, TIMESTAMP_COL]
    built = {
        "Clean": (context, forecast),
        "Gaps as NaN": (as_nan, forecast),
        "Rows dropped": (dropped, forecast),
        "Spike": (spiked, forecast),
    }
    return built, actuals, spike_at


def test_plot_scenarios_smoke(scenarios, tmp_path: Path) -> None:
    built, actuals, spike_at = scenarios
    fig = plotting.plot_scenarios(
        built,
        actuals=actuals,
        spikes={"Spike": spike_at},
        notes={"Clean": "MAE 0"},
        title="Scenarios",
    )
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in visible] == list(built)
    assert visible[0].get_title(loc="right") == "MAE 0"
    assert len({ax.get_ylim() for ax in visible}) == 1, "panels must share the y scale"
    labels = legend_labels(fig.legends[0])
    assert labels[:3] == ["History", "Actual", "Forecast (median)"]
    assert labels[-2:] == ["Missing context", "Spike"]
    assert fig.legends[0].get_title().get_text() == "Scenarios"
    assert_no_dual_axes(fig)
    # One gray span per gap, for NaN values and dropped rows alike; none when clean.
    assert [len(ax.patches) for ax in visible] == [0, 2, 2, 0]
    assert_saved(fig, tmp_path, "scenarios")


def test_plot_scenarios_grid_hides_empty_slots(scenarios) -> None:
    built, actuals, _ = scenarios
    three = dict(list(built.items())[:3])
    fig = plotting.plot_scenarios(three, actuals=actuals, ncols=2, sharey=False)
    assert [ax.get_visible() for ax in fig.axes] == [True, True, True, False]
    above_empty = fig.axes[1]
    assert above_empty.xaxis.get_tick_params()["labelbottom"]


def test_plot_scenarios_validates_keys_and_spikes(scenarios) -> None:
    built, _, spike_at = scenarios
    with pytest.raises(ValueError, match="spikes keys"):
        plotting.plot_scenarios(built, spikes={"Unknown": spike_at})
    with pytest.raises(ValueError, match="no row in the plotted history"):
        plotting.plot_scenarios(built, spikes={"Spike": pd.Timestamp("2000-01-01")})
    with pytest.raises(ValueError, match="at least one"):
        plotting.plot_scenarios({})


def test_plot_scenarios_spikes_accept_datetimes_and_reject_positions(scenarios) -> None:
    built, _, spike_at = scenarios
    for spike in (spike_at.to_pydatetime(), [np.datetime64(spike_at)], str(spike_at)):
        fig = plotting.plot_scenarios(built, spikes={"Spike": spike})
        assert legend_labels(fig.legends[0])[-1] == "Spike"
    for position in (290, [290]):
        with pytest.raises(TypeError, match=r"iloc\[pos\]"):
            plotting.plot_scenarios(built, spikes={"Spike": position})


def test_plot_scenarios_shades_gaps_at_the_window_start_in_a_shared_time_range(series) -> None:
    context, forecast, _ = series
    n, history = len(context), 168
    first_hours = np.arange(n - history, n - history + 6)
    as_nan = context.copy()
    as_nan.loc[first_hours, TARGET_COL] = np.nan
    dropped = context.drop(index=first_hours).reset_index(drop=True)
    longer = forecast_for(context, horizon=2 * HORIZON)
    fig = plotting.plot_scenarios(
        {"Gaps as NaN": (as_nan, forecast), "Rows dropped": (dropped, longer)}, history=history
    )
    window_start = context.loc[n - history, TIMESTAMP_COL]
    half = pd.Timedelta(minutes=30)
    for ax in fig.axes:
        (span,) = ax.patches  # the same gap, whether NaN or dropped
        assert span.get_x() == pytest.approx(mdates.date2num(window_start - half))
        span_end = window_start + pd.Timedelta(hours=5) + half
        assert span.get_x() + span.get_width() == pytest.approx(mdates.date2num(span_end))
        start, end = ax.get_xlim()
        assert start == pytest.approx(mdates.date2num(window_start))
        assert end == pytest.approx(mdates.date2num(longer[TIMESTAMP_COL].max()))


def test_plot_scenarios_shades_only_real_gaps_for_calendar_frequencies() -> None:
    business = frame_on(pd.date_range("2015-01-01", periods=120, freq="B"))
    forecast = forecast_for(business, horizon=10, freq="B")
    gappy = business.drop(index=[100, 101]).reset_index(drop=True)
    fig = plotting.plot_scenarios(
        {"Clean": (business, forecast), "Dropped": (gappy, forecast)}, freq="B", history=48
    )
    assert [len(ax.patches) for ax in fig.axes] == [0, 1], "weekends are not gaps"
    monthly = frame_on(pd.date_range("2010-01-01", periods=60, freq="MS"))
    fig = plotting.plot_scenarios({"Clean": (monthly, forecast_for(monthly, freq="MS"))})
    assert len(fig.axes[0].patches) == 0
    assert "Missing context" not in legend_labels(fig.legends[0])


def test_missing_spans_cover_nan_values_and_absent_rows() -> None:
    index = pd.date_range("2018-01-01", periods=10, freq="h")
    values = pd.Series(np.arange(10.0), index=index)
    values.iloc[2:4] = np.nan
    gappy = values.drop(index[[0, 7]])  # a dropped row at the window start, one inside
    spans = plotting._missing_spans(gappy, index[0], index[-1], to_offset("h"))
    half = pd.Timedelta(minutes=30)
    assert spans == [
        (index[0] - half, index[0] + half),
        (index[2] - half, index[3] + half),
        (index[7] - half, index[7] + half),
    ]


# --- plot_gap_grid ------------------------------------------------------------------------


def test_plot_gap_grid_smoke(tmp_path: Path) -> None:
    full = hourly_frame(24 * 7)
    before = full.drop(index=range(30, 36)).reset_index(drop=True)
    before.loc[[50, 51], TARGET_COL] = np.nan
    after = full.copy()
    after.loc[range(30, 36), TARGET_COL] = np.nan
    after.loc[after[TIMESTAMP_COL].isin(before.loc[[50, 51], TIMESTAMP_COL]), TARGET_COL] = np.nan
    fig = plotting.plot_gap_grid(before, after, column=TARGET_COL, title="Gaps")
    left, right = fig.axes
    missing = [int((ax.collections[0].get_array() == 2).sum()) for ax in (left, right)]
    assert missing == [6, 8]
    assert left.get_title(loc="left") == "Before: 6 rows missing"
    assert right.get_title(loc="left").startswith("After: 8 values NaN")
    assert legend_labels(fig.legends[0]) == ["Observed", "Missing"]
    assert left.collections[0].get_linewidth()[0] > 0, "a week keeps its white cell separators"
    assert_no_dual_axes(fig)
    assert_saved(fig, tmp_path, "gap_grid")


def test_plot_gap_grid_draws_every_cell_of_a_full_year() -> None:
    full = hourly_frame(24 * 365, start="2018-01-01")
    after = full.copy()
    after.loc[[100, 2_000, 5_000, 8_000], TARGET_COL] = np.nan
    fig = plotting.plot_gap_grid(full, after, column=TARGET_COL)
    left, right = fig.axes
    # Thin rows drop the white separators, which used to cover the cells entirely.
    assert left.collections[0].get_linewidth()[0] == 0
    assert pixel_share(plot_area_pixels(fig, left), plotting.OBSERVED_CELL) > 0.95
    right_pixels = plot_area_pixels(fig, right)
    assert pixel_share(right_pixels, plotting.OBSERVED_CELL) > 0.9
    assert pixel_share(right_pixels, plotting.ISSUE_COLOR) * right_pixels[..., 0].size > 4 * 40


def test_plot_gap_grid_needs_max_days_for_a_multi_year_overview() -> None:
    full = hourly_frame(24 * 1461, start="2015-01-01")
    after = full.copy()
    after.loc[[500, 20_000], TARGET_COL] = np.nan
    with pytest.raises(ValueError, match="max_days=400"):
        plotting.plot_gap_grid(full, after, column=TARGET_COL)
    fig = plotting.plot_gap_grid(full, after, column=TARGET_COL, max_days=1500)
    left, right = fig.axes
    assert pixel_share(plot_area_pixels(fig, left), plotting.OBSERVED_CELL) > 0.95
    assert pixel_share(plot_area_pixels(fig, right), plotting.ISSUE_COLOR) > 0


def test_plot_gap_grid_anchors_the_grid_at_the_series_start() -> None:
    odd_hours = frame_on(pd.date_range("2018-03-01 01:00", periods=48, freq="2h"))
    before = odd_hours.drop(index=[3, 4]).reset_index(drop=True)
    fig = plotting.plot_gap_grid(before, odd_hours, column=TARGET_COL, freq="2h")
    left, right = fig.axes
    assert left.get_title(loc="left") == "Before: 2 rows missing"
    assert right.get_title(loc="left").startswith("After: 0 values NaN")
    assert left.get_xticklabels()[0].get_text() == "01:00"


def test_plot_gap_grid_rejects_off_grid_timestamps() -> None:
    frame = hourly_frame(48)
    off_grid = frame.copy()
    off_grid.loc[5, TIMESTAMP_COL] += pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="off the 'h' grid"):
        plotting.plot_gap_grid(off_grid, frame, column=TARGET_COL)


@pytest.mark.parametrize(("freq", "message"), [("7h", "divide one day"), ("MS", "fixed duration")])
def test_plot_gap_grid_rejects_unusable_freq(freq: str, message: str) -> None:
    frame = hourly_frame(48)
    with pytest.raises(ValueError, match=message):
        plotting.plot_gap_grid(frame, frame, column=TARGET_COL, freq=freq)


def test_plot_gap_grid_rejects_non_positive_max_days() -> None:
    frame = hourly_frame(48)
    with pytest.raises(ValueError, match="max_days must be positive"):
        plotting.plot_gap_grid(frame, frame, column=TARGET_COL, max_days=0)


# --- plot_benchmarks ----------------------------------------------------------------------


def test_paper_benchmarks_csv_matches_the_paper_tables() -> None:
    df = pd.read_csv(BENCHMARKS_CSV)
    assert list(df.columns) == ["model", "benchmark", "skill_score_pct", "source"]
    assert df.shape == (18, 4)
    assert not df.duplicated(["model", "benchmark"]).any()
    chronos2 = df.loc[df["model"] == "Chronos-2"].set_index("benchmark")["skill_score_pct"]
    assert chronos2.to_dict() == {
        "fev-bench": 47.3,
        "GIFT-Eval": 51.4,
        "Chronos Benchmark II": 46.6,
    }
    assert set(df["source"]) == {"Ansari et al. (2025), Chronos-2 technical report, Tables 3-5"}


def test_plot_benchmarks_from_csv(tmp_path: Path) -> None:
    df = pd.read_csv(BENCHMARKS_CSV)
    fig = plotting.plot_benchmarks(df, title="Benchmarks")
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in axes] == [
        "fev-bench",
        "GIFT-Eval",
        "Chronos Benchmark II",
    ]
    for ax in axes:
        bars = ax.patches
        assert len(bars) == 6
        top = max(bars, key=lambda bar: bar.get_y())
        assert to_hex(top.get_facecolor()) == plotting.BLUE  # Chronos-2 is first everywhere
        others = {to_hex(bar.get_facecolor()) for bar in bars if bar is not top}
        assert others == {plotting.NEUTRAL_BAR}
        widths = [bar.get_width() for bar in sorted(bars, key=lambda bar: bar.get_y())]
        assert widths == sorted(widths), "bars must be sorted within each panel"
        emphasis = {
            label.get_text(): (to_hex(label.get_color()), label.get_fontweight())
            for label in ax.get_yticklabels()
        }
        assert emphasis["Chronos-2"] == (plotting.INK_PRIMARY, "bold")
        assert emphasis["TiRex"] == (plotting.INK_SECONDARY, "normal")
    labels = {text.get_text() for text in axes[0].texts}
    assert {"47.3", "42.6", "38.9"} <= labels
    assert_no_dual_axes(fig)
    assert_saved(fig, tmp_path, "benchmarks")


def test_plot_benchmarks_validates_input() -> None:
    df = pd.read_csv(BENCHMARKS_CSV)
    with pytest.raises(ValueError, match="highlight 'Nope'"):
        plotting.plot_benchmarks(df, highlight="Nope")
    with pytest.raises(ValueError, match="duplicated"):
        plotting.plot_benchmarks(pd.concat([df, df.head(1)]))
    with pytest.raises(ValueError, match="missing columns"):
        plotting.plot_benchmarks(df.drop(columns="benchmark"))
    with pytest.raises(ValueError, match="ncols"):
        plotting.plot_benchmarks(df, ncols=0)


def test_plot_benchmarks_grid_layout_and_no_highlight() -> None:
    df = pd.read_csv(BENCHMARKS_CSV)
    fig = plotting.plot_benchmarks(df, highlight=None, ncols=2)
    assert [ax.get_visible() for ax in fig.axes] == [True, True, True, False]
    colors = {to_hex(bar.get_facecolor()) for ax in fig.axes for bar in ax.patches}
    assert colors == {plotting.NEUTRAL_BAR}


# --- plot_metric_vs_param -----------------------------------------------------------------


def metric_frame(models: list[str]) -> pd.DataFrame:
    lengths = [256, 512, 1024, 2048]
    rows = [
        {"model": model, "context_length": length, "WQL": 0.1 - 0.01 * i + 0.002 * j}
        for i, model in enumerate(models)
        for j, length in enumerate(lengths)
    ]
    frame = pd.DataFrame(rows)
    return frame.assign(low=frame["WQL"] - 0.003, high=frame["WQL"] + 0.003)


def test_plot_metric_vs_param_smoke(tmp_path: Path) -> None:
    models = ["chronos-2", "seasonal_naive", "auto_ets"]
    ax = plotting.plot_metric_vs_param(
        metric_frame(models),
        x="context_length",
        y="WQL",
        hue="model",
        band=("low", "high"),
        logx=True,
        title="Error vs context length",
    )
    assert legend_labels(ax.get_legend()) == models
    assert [to_hex(line.get_color()) for line in ax.lines] == [
        plotting.MODEL_COLORS[model] for model in models
    ]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["256", "512", "1,024", "2,048"]
    assert ax.get_xscale() == "log"
    assert_no_dual_axes(ax.figure)
    assert_saved(ax.figure, tmp_path, "metric_vs_param")


def test_plot_metric_vs_param_single_series_has_no_legend() -> None:
    ax = plotting.plot_metric_vs_param(
        metric_frame(["chronos-2"]), x="context_length", y="WQL", hue="model", title="One line"
    )
    assert ax.get_legend() is None
    assert ax.get_title(loc="left") == "One line"


def test_plot_metric_vs_param_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        plotting.plot_metric_vs_param(metric_frame(["chronos-2"]), x="nope", y="WQL", hue="model")


def test_plot_metric_vs_param_rejects_nan_hue() -> None:
    frame = metric_frame(["chronos-2", "auto_ets"])
    frame.loc[0, "model"] = np.nan  # groupby would drop it and shift every legend label
    with pytest.raises(ValueError, match="'model'"):
        plotting.plot_metric_vs_param(frame, x="context_length", y="WQL", hue="model")


def test_plot_metric_vs_param_wraps_a_wide_legend_in_reading_order() -> None:
    models = [
        "chronos-2",
        "seasonal_naive",
        "auto_ets",
        "chronos-bolt",
        "chronos-2-small",
        "chronos-2-synth",
    ]
    kwargs = {"x": "context_length", "y": "WQL", "hue": "model", "logx": True}
    narrow = plotting.plot_metric_vs_param(metric_frame(models[:2]), **kwargs)
    wide = plotting.plot_metric_vs_param(metric_frame(models), title="Six models", **kwargs)
    for ax in (narrow, wide):
        ax.figure.canvas.draw()
    # The legend wraps instead of making constrained layout squeeze the plot.
    assert wide.get_position().width == pytest.approx(narrow.get_position().width, abs=0.02)
    legend = wide.get_legend()
    assert len({round(text.get_window_extent().y0) for text in legend.get_texts()}) > 1
    assert reading_order(legend.get_texts()) == models
    expected = plotting.assign_colors(models)
    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True):
        assert to_hex(handle.get_color()) == expected[text.get_text()]
    (title,) = [text for text in wide.findobj(Text) if text.get_text() == "Six models"]
    assert title.get_window_extent().y0 >= legend.get_window_extent().y1, "title clears legend"
