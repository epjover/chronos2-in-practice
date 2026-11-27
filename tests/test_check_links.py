"""Offline tests of the link checker: collection, repository links, titles and verdicts.

The suite runs without a network, so the requests are answered by a stub: these tests cover
what is requested, how each answer is classified and what the command prints and returns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import requests

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


class Answer:
    """The parts of a ``requests`` response that the checker reads."""

    def __init__(self, status: int, *, body: str = "", url: str = "", location: str = "") -> None:
        self.status_code = status
        self.text = body
        self.url = url
        self.headers = {"content-type": "text/html", "location": location}


def answer_with(monkeypatch: pytest.MonkeyPatch, pages: dict[str, Answer]) -> None:
    """Serve ``pages`` to the checker; any other URL fails like a host that does not answer."""

    def get(url: str, **_: object) -> Answer:
        if url not in pages:
            raise requests.ConnectionError(f"no route to {url}")
        page = pages[url]
        page.url = page.url or url
        return page

    monkeypatch.setattr(links.requests, "get", get)
    monkeypatch.setattr(links.requests, "head", get)


def test_check_url_classifies_each_kind_of_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = links.HOST_CONTROLS["www.kaggle.com"]
    answer_with(
        monkeypatch,
        {
            "https://a.org/page": Answer(200, body="<title>A page</title>"),
            "https://a.org/moved": Answer(200, body="<title>New</title>", url="https://a.org/new"),
            "https://a.org/gone": Answer(404),
            "https://a.org/browsers-only": Answer(403),
            "https://a.org/accepted": Answer(202, body="  "),
            "https://www.kaggle.com/datasets/x/y": Answer(404),
            control: Answer(404),
            "https://doi.org/10.1/ok": Answer(302, location="https://publisher.org/paper"),
            "https://doi.org/10.1/unknown": Answer(404),
        },
    )

    def verdict(url: str) -> tuple[str, str]:
        result = links.check_url(url, {"02", "01"}, tmp_path)
        assert result.articles == "01,02"
        return result.verdict, result.status

    assert verdict("https://a.org/page") == ("ok", "200")
    assert links.check_url("https://a.org/page", {"01"}, tmp_path).title == "A page"
    assert links.check_url("https://a.org/moved", {"01"}, tmp_path).final_url == "https://a.org/new"
    assert verdict("https://a.org/gone") == ("broken", "404")
    assert verdict("https://a.org/browsers-only") == ("refused", "403")
    # an empty "202 Accepted" is not a page: it must not pass as a working link
    assert verdict("https://a.org/accepted") == ("refused", "202 with an empty body")
    # a host that answers 404 for a page that certainly exists hides every page from scripts
    assert verdict("https://www.kaggle.com/datasets/x/y") == ("refused", "404 for any page")
    assert verdict("https://doi.org/10.1/ok") == ("ok", "302 resolves")
    assert links.check_url("https://doi.org/10.1/ok", {"01"}, tmp_path).final_url.endswith("paper")
    assert verdict("https://doi.org/10.1/unknown")[0] == "broken"
    assert verdict("https://nowhere.invalid/x") == ("broken", "ConnectionError")
    assert verdict(f"https://github.com/{links.GITHUB_REPO}/blob/main/missing.py")[0] == "broken"


def test_a_failing_control_page_does_not_excuse_other_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = links.HOST_CONTROLS["www.kaggle.com"]
    answer_with(monkeypatch, {control: Answer(404)})
    assert links.same_failure_on_control("https://www.kaggle.com/datasets/x/y", 404)
    assert not links.same_failure_on_control("https://www.kaggle.com/datasets/x/y", 500)
    assert not links.same_failure_on_control(control, 404)  # the control cannot vouch for itself
    assert not links.same_failure_on_control("https://a.org/gone", 404)
    answer_with(monkeypatch, {})
    assert not links.same_failure_on_control("https://www.kaggle.com/datasets/x/y", 404)


def test_main_lists_broken_links_first_and_fails_on_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    articles = tmp_path / "articles"
    (articles / "medium").mkdir(parents=True)
    (articles / "01-a.md").write_text("https://a.org/page and https://a.org/gone")
    monkeypatch.setattr(links, "find_repo_root", lambda: tmp_path)
    answer_with(
        monkeypatch,
        {
            "https://a.org/page": Answer(200, body="<title>A page</title>"),
            "https://a.org/gone": Answer(404),
        },
    )
    table = tmp_path / "links.csv"
    assert links.main(["--csv", str(table)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("broken") and "https://a.org/gone" in lines[0]
    assert lines[1].startswith("ok") and lines[1].endswith("A page")
    assert lines[-1].startswith("2 links")
    assert table.read_text().count("\n") == 3

    (articles / "01-a.md").write_text("https://a.org/page")
    assert links.main([]) == 0
