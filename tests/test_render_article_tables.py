"""Tests for ``scripts/render_article_tables.py`` (Markdown tables -> PNG + Medium copies)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_article_tables.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_article_tables", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations via sys.modules
    spec.loader.exec_module(module)
    return module


rat = _load_script()

ARTICLE = """# Title

Some prose with a [link](../results/figures/x.png).

| Model | WQL | Skill vs naive |
| --- | ---: | --- |
| `chronos-2` | **0.0388** | 63.9% [60.4, 66.9] |
| seasonal_naive | 0.0748 | n/a |

![Figure 1: results/figures/01_fan_chart.png](../results/figures/01_fan_chart.png)

| Check | Why |
| --- | --- |
| Regular grid | A missing row shifts everything \\| no warning |
"""


def test_parse_cell_resolves_inline_markdown() -> None:
    assert rat.parse_cell(" **0.0388** ") == rat.Cell("0.0388", bold=True, code=False)
    leading = rat.parse_cell("**63.9%** [60.4, 66.9]")
    assert leading == rat.Cell("63.9% [60.4, 66.9]", bold=True, code=False)
    assert not rat.parse_cell("plain, then **bold**").bold
    assert rat.parse_cell("`chronos-2`") == rat.Cell("chronos-2", bold=False, code=True)
    assert rat.parse_cell("63.9% [60.4, 66.9]").text == "63.9% [60.4, 66.9]"
    assert rat.parse_cell("*Figure* text").text == "Figure text"
    assert rat.parse_cell("[name](http://x)").text == "name"


def test_split_row_and_alignments() -> None:
    assert rat.split_row("| a | b \\| c |") == [" a ", " b | c "]
    assert rat.alignments("| --- | ---: | :---: |") == ["left", "right", "center"]


def test_wrap_cell_wraps_prose_but_not_code() -> None:
    long = rat.Cell("word " * 30, bold=False, code=False)
    assert len(rat.wrap_cell(long)) > 1
    code = rat.Cell("x" * 80, bold=False, code=True)
    assert rat.wrap_cell(code) == ["x" * 80]


def test_convert_article_replaces_tables_with_images(tmp_path: Path) -> None:
    source = tmp_path / "articles" / "07-test-article.md"
    source.parent.mkdir()
    source.write_text(ARTICLE, encoding="utf-8")
    out_dir = tmp_path / "articles" / "medium"
    target, n_tables = rat.convert_article(source, out_dir)
    assert n_tables == 2
    text = target.read_text(encoding="utf-8")
    assert "| Model |" not in text
    assert "![Table 1: articles/medium/tables/07_table_1.png](tables/07_table_1.png)" in text
    assert "*Table 1. Columns: Model, WQL, Skill vs naive.*" in text
    assert "![Table 2: articles/medium/tables/07_table_2.png](tables/07_table_2.png)" in text
    # figures keep a relative path, one level deeper; other links go to GitHub
    assert "](../../results/figures/01_fan_chart.png)" in text
    assert f"[link]({rat.REPO_URL}/blob/main/results/figures/x.png)" in text
    for k in (1, 2):
        image = out_dir / "tables" / f"07_table_{k}.png"
        assert image.is_file() and image.stat().st_size > 5_000


def test_absolute_links_tell_directories_from_files(tmp_path: Path) -> None:
    (tmp_path / "bench").mkdir()
    text = (
        "See [`bench/`](../bench/), [the script](../scripts/cost_table.py) and "
        "![Figure 1: x](../results/figures/x.png). [Elsewhere](https://example.org) stays."
    )
    converted = rat.absolute_links(text, tmp_path)
    assert f"[`bench/`]({rat.REPO_URL}/tree/main/bench)" in converted
    assert f"[the script]({rat.REPO_URL}/blob/main/scripts/cost_table.py)" in converted
    assert "![Figure 1: x](../results/figures/x.png)" in converted
    assert "[Elsewhere](https://example.org)" in converted


def test_link_parts_fills_published_parts_and_reports_the_rest() -> None:
    text = "[Part 1](#) set it up and [Part 2](#) measured it."
    filled, pending = rat.link_parts(text, {"1": "https://medium.com/@someone/part-1"})
    assert filled.startswith("[Part 1](https://medium.com/@someone/part-1) set it up")
    assert filled.endswith("[Part 2](#) measured it.")
    assert pending == ["2"]


def test_load_part_urls_ignores_empty_and_rejects_non_https(tmp_path: Path) -> None:
    articles = tmp_path / "articles"
    articles.mkdir()
    assert rat.load_part_urls(tmp_path) == {}
    (articles / rat.PART_URLS_FILE).write_text('{"1": "https://medium.com/p/1", "2": ""}')
    assert rat.load_part_urls(tmp_path) == {"1": "https://medium.com/p/1"}
    (articles / rat.PART_URLS_FILE).write_text('{"1": "medium.com/p/1"}')
    with pytest.raises(ValueError, match="https://"):
        rat.load_part_urls(tmp_path)
