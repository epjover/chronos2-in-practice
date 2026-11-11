"""Offline tests of the link checker: collection, repository links and page titles.

The requests themselves are not tested here (the suite runs without a network); these cover
the parts that decide *what* is requested and how a repository link is judged.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_links.py"
spec = importlib.util.spec_from_file_location("check_links", SCRIPT)
links = importlib.util.module_from_spec(spec)
sys.modules["check_links"] = links
spec.loader.exec_module(links)


def test_collect_urls_reads_drafts_and_medium_copies(tmp_path: Path) -> None:
    articles = tmp_path / "articles"
    (articles / "medium").mkdir(parents=True)
    (articles / "01-a.md").write_text("See https://example.org/a. And (https://example.org/b).")
    (articles / "medium" / "02-b.md").write_text(
        "[x](https://example.org/a) `https://example.org/c`"
    )
    (articles / "notes.md").write_text("https://example.org/ignored")
    found = links.collect_urls(tmp_path)
    assert found == {
        "https://example.org/a": {"01", "02"},
        "https://example.org/b": {"01"},
        "https://example.org/c": {"02"},
    }


def test_repository_links_are_checked_in_the_working_tree(tmp_path: Path) -> None:
    (tmp_path / "bench").mkdir()
    (tmp_path / "bench" / "run.py").write_text("")
    repo = f"https://github.com/{links.GITHUB_REPO}"
    assert links.check_repo_link(repo, tmp_path) == ("ok", "in the working tree")
    assert links.check_repo_link(f"{repo}/tree/main/bench", tmp_path)[0] == "ok"
    assert links.check_repo_link(f"{repo}/blob/main/bench/run.py", tmp_path)[0] == "ok"
    # a directory is not a file, and a missing path is broken
    assert links.check_repo_link(f"{repo}/blob/main/bench", tmp_path)[0] == "broken"
    assert links.check_repo_link(f"{repo}/blob/main/nope.py", tmp_path)[0] == "broken"
    assert links.check_repo_link("https://github.com/other/repo", tmp_path) is None


def test_page_title_is_cleaned() -> None:
    body = "<html><head><title>\n  Chronos-2 &amp; friends  </title></head></html>"
    assert links.page_title(body) == "Chronos-2 & friends"
    assert links.page_title("no title here") == ""
