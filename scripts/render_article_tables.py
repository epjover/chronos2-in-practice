"""Render every Markdown table of the articles as a PNG and write Medium-ready copies.

Medium's editor has no table block: it accepts text, images, code blocks and embeds, so a
Markdown table pasted into it collapses into one line per cell. This script keeps the tables in
``articles/*.md`` as the source of truth and produces what the editor can show:

- ``articles/medium/tables/<article>_table_<k>.png``: one image per table, drawn in the series'
  figure style (same fonts and inks as ``results/figures``), with bold cells, code cells and
  column alignment preserved and long cells wrapped.
- ``articles/medium/<article>.md``: the article with each table replaced by the image and a
  caption ("Table k. <column headers>"), ready to paste or import. In this copy every relative
  link to a file of the repository becomes a GitHub URL (a relative link means nothing on
  Medium), and each ``[Part N](#)`` cross-reference takes the address of that part from
  ``articles/medium_urls.json`` once it has one. Figures keep their relative path: they are
  uploaded to Medium by hand.

Run it after editing an article; the images are regenerated from scratch each time.

Usage::

    uv run python scripts/render_article_tables.py            # all five articles
    uv run python scripts/render_article_tables.py --only 05  # one article
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c2p.config import GITHUB_REPO, find_repo_root
from c2p.plotting import GRID_COLOR, INK_PRIMARY, INK_SECONDARY, style_context

TABLE_RE = re.compile(r"(?m)^(\|[^\n]*\|)\n(\|[ :\-|]+\|)\n((?:\|[^\n]*\|\n?)+)")
MONO_FONT = "DejaVu Sans Mono"
WRAP_AT = 42  # characters per line inside a cell before wrapping
MAX_WIDTH_IN = 9.0  # inches; Medium renders images up to about 1,400 px wide
DPI = 200
REPO_URL = f"https://github.com/{GITHUB_REPO}"
PART_URLS_FILE = "medium_urls.json"  # under articles/: {"1": "https://medium.com/...", ...}
RELATIVE_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\((\.\./[^)#\s]*)\)")
PART_LINK_RE = re.compile(r"\[Part (\d)\]\(#\)")


@dataclass(frozen=True)
class Cell:
    """One table cell with its inline Markdown resolved."""

    text: str
    bold: bool
    code: bool


def parse_cell(raw: str) -> Cell:
    """Strip ``**``, backticks and ``*`` from a cell; remember if it was bold or code.

    A cell that opens with a bold run (``**63.9%** [60.4, 66.9]``) is drawn bold in full: the
    emphasis marks the cell, and a table image has no partial weights.
    """
    text = raw.strip()
    bold = re.match(r"\*\*.+?\*\*", text) is not None
    code = text.startswith("`") and text.endswith("`") and text.count("`") == 2
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return Cell(text=text, bold=bold, code=code)


def split_row(line: str) -> list[str]:
    """Split a ``| a | b |`` line into cells, honouring escaped pipes."""
    inner = line.strip().strip("|")
    return [part.replace("\\|", "|") for part in re.split(r"(?<!\\)\|", inner)]


def alignments(separator: str) -> list[str]:
    """``left``/``right``/``center`` per column from the ``| --- | ---: |`` line."""
    out = []
    for raw in split_row(separator):
        part = raw.strip()
        if part.startswith(":") and part.endswith(":"):
            out.append("center")
        elif part.endswith(":"):
            out.append("right")
        else:
            out.append("left")
    return out


def wrap_cell(cell: Cell, width: int = WRAP_AT) -> list[str]:
    """Lines of a cell after wrapping; code cells are never wrapped."""
    if cell.code or len(cell.text) <= width:
        return [cell.text or " "]
    return textwrap.wrap(cell.text, width=width, break_long_words=False) or [" "]


def render_table(header: list[Cell], rows: list[list[Cell]], aligns: list[str], path: Path) -> Path:
    """Draw one table to ``path`` and return it."""
    n_cols = len(header)
    grid = [header, *rows]
    wrapped = [[wrap_cell(cell) for cell in row] for row in grid]
    col_chars = [
        max(max(len(line) for line in wrapped[r][c]) for r in range(len(grid)))
        for c in range(n_cols)
    ]
    # Column widths in inches: proportional to the longest line, with a floor for short columns.
    widths = [max(0.9, 0.085 * chars + 0.35) for chars in col_chars]
    scale = min(1.0, MAX_WIDTH_IN / sum(widths))
    widths = [w * scale for w in widths]
    row_lines = [max(len(lines) for lines in row) for row in wrapped]
    line_h = 0.21 * (1.0 if scale == 1.0 else max(0.85, scale))
    pad = 0.12
    row_h = [lines * line_h + 2 * pad for lines in row_lines]
    fig_w, fig_h = sum(widths), sum(row_h) + 0.08
    font = 9.5 if scale == 1.0 else max(7.5, 9.5 * scale)

    with style_context():
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, fig_w)
        ax.set_ylim(0, fig_h)
        ax.axis("off")
        y_top = fig_h - 0.04
        for r, row in enumerate(wrapped):
            y0 = y_top - row_h[r]
            if r == 0:
                ax.add_patch(plt.Rectangle((0, y0), fig_w, row_h[r], color="#f3f2ee", linewidth=0))
            elif r % 2 == 0:
                ax.add_patch(plt.Rectangle((0, y0), fig_w, row_h[r], color="#faf9f6", linewidth=0))
            x = 0.0
            for c in range(n_cols):
                cell = grid[r][c]
                lines = row[c]
                align = "left" if r == 0 and aligns[c] != "right" else aligns[c]
                if align == "right":
                    tx, ha = x + widths[c] - 0.12, "right"
                elif align == "center":
                    tx, ha = x + widths[c] / 2, "center"
                else:
                    tx, ha = x + 0.12, "left"
                ax.text(
                    tx,
                    y0 + row_h[r] / 2,
                    "\n".join(lines),
                    ha=ha,
                    va="center_baseline",  # same baseline along a row, whatever the glyphs
                    fontsize=font,
                    fontweight="bold" if (r == 0 or cell.bold) else "normal",
                    family=MONO_FONT if cell.code else None,
                    color=INK_PRIMARY if (r == 0 or cell.bold) else INK_SECONDARY,
                    linespacing=1.25,
                )
                x += widths[c]
            if r == 0:
                ax.plot([0, fig_w], [y0, y0], color=INK_SECONDARY, linewidth=1.0)
            else:
                ax.plot([0, fig_w], [y0, y0], color=GRID_COLOR, linewidth=0.6)
            y_top = y0
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI, facecolor="white")
        plt.close(fig)
    return path


def absolute_links(text: str, root: Path) -> str:
    """Point every relative, non-image link of an article at the file on GitHub.

    Links are resolved from ``articles/``; a directory becomes a ``tree`` URL and a file a
    ``blob`` URL. Image links are left alone.
    """

    def replace(match: re.Match[str]) -> str:
        path = posixpath.normpath(posixpath.join("articles", match.group(2)))
        kind = "tree" if (root / path).is_dir() else "blob"
        return f"[{match.group(1)}]({REPO_URL}/{kind}/main/{path})"

    return RELATIVE_LINK_RE.sub(replace, text)


def load_part_urls(root: Path) -> dict[str, str]:
    """Published address of each part, from ``articles/medium_urls.json`` (empty if not yet)."""
    path = root / "articles" / PART_URLS_FILE
    if not path.is_file():
        return {}
    urls = json.loads(path.read_text(encoding="utf-8"))
    bad = {part: url for part, url in urls.items() if url and not url.startswith("https://")}
    if bad:
        raise ValueError(f"{path.name}: addresses must start with https://, got {bad}.")
    return {str(part): url for part, url in urls.items() if url}


def link_parts(text: str, urls: Mapping[str, str]) -> tuple[str, list[str]]:
    """Fill ``[Part N](#)`` from ``urls``; return the text and the parts that have no address."""
    pending: list[str] = []

    def replace(match: re.Match[str]) -> str:
        part = match.group(1)
        if part in urls:
            return f"[Part {part}]({urls[part]})"
        pending.append(part)
        return match.group(0)

    return PART_LINK_RE.sub(replace, text), pending


def convert_article(
    source: Path,
    out_dir: Path,
    *,
    root: Path | None = None,
    part_urls: Mapping[str, str] | None = None,
) -> tuple[Path, int]:
    """Write the Medium copy of ``source`` with its tables as images; return (path, n_tables).

    ``root`` is the repository root used to tell directories from files when links are made
    absolute (default: two levels above ``out_dir``). ``part_urls`` fills the ``[Part N](#)``
    cross-references; parts without an address keep the placeholder.
    """
    text = source.read_text(encoding="utf-8")
    root = root if root is not None else out_dir.parents[1]
    stem = source.stem[:2]
    tables_dir = out_dir / "tables"
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        header = [parse_cell(c) for c in split_row(match.group(1))]
        aligns = alignments(match.group(2))
        rows = [
            [parse_cell(c) for c in split_row(line)]
            for line in match.group(3).strip("\n").split("\n")
        ]
        n = len(header)
        rows = [
            row + [Cell(" ", False, False)] * (n - len(row)) if len(row) < n else row[:n]
            for row in rows
        ]
        image = render_table(header, rows, aligns, tables_dir / f"{stem}_table_{counter}.png")
        rel = image.relative_to(out_dir).as_posix()
        headers = ", ".join(c.text for c in header if c.text.strip())
        caption = f"*Table {counter}. Columns: {headers}.*"
        return f"![Table {counter}: articles/medium/{rel}]({rel})\n\n{caption}\n"

    converted = TABLE_RE.sub(replace, text)
    converted = absolute_links(converted, root)
    converted, _ = link_parts(converted, part_urls or {})
    # Figures live in results/figures; from articles/medium that is one level deeper.
    converted = re.sub(r"(!\[[^\]]*\]\()\.\./", r"\1../../", converted)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / source.name
    target.write_text(converted, encoding="utf-8")
    return target, counter


def main(argv: Sequence[str] | None = None) -> int:
    """Render the tables of every article (or of ``--only NN``) and write the Medium copies."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default=None, help="article number prefix, e.g. 05")
    args = parser.parse_args(argv)
    root = find_repo_root()
    articles = sorted((root / "articles").glob("0*.md"))
    if args.only:
        articles = [a for a in articles if a.name.startswith(args.only)]
    if not articles:
        raise SystemExit("No articles found.")
    out_dir = root / "articles" / "medium"
    part_urls = load_part_urls(root)
    total = 0
    pending: dict[str, list[str]] = {}
    for article in articles:
        target, n = convert_article(article, out_dir, root=root, part_urls=part_urls)
        total += n
        print(f"{article.name}: {n} tables -> {target.relative_to(root)}")
        _, missing = link_parts(article.read_text(encoding="utf-8"), part_urls)
        if missing:
            pending[article.name[:2]] = missing
    print(f"{total} tables rendered to {out_dir.relative_to(root) / 'tables'}")
    for article, parts in pending.items():
        print(
            f"Part {article[1]}: link to Part {', '.join(parts)} still points to '#'; add its "
            f"address to articles/{PART_URLS_FILE} once it is published."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
