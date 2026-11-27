"""Request every link of the article drafts once and report which ones answer.

Run it before publishing: a reference that moved is cheap to fix in a draft and awkward to fix
in a published article. For each distinct URL of ``articles/*.md`` and ``articles/medium/*.md``
the script records the HTTP status, the address after redirects and the page title.

Links into this repository are checked against the working tree instead of GitHub (a ``blob``
link must be a file, a ``tree`` link a directory), so the check also works while the repository
is private. A 401, 403 or 429 is reported as *refused*, not as broken: some publishers turn
scripts away and still serve the page to a browser. Two cases get their own rule:

- a ``doi.org`` link is checked at the resolver, without following it to the publisher: a DOI
  that resolves is a working reference, whatever the publisher's site says to a script;
- a host listed in ``HOST_CONTROLS`` answers scripts with an error for every address. The link is
  *refused*, not broken, when a page of that host known to exist fails in the same way.

Usage::

    uv run python scripts/check_links.py                 # exit code 1 if a link is broken
    uv run python scripts/check_links.py --csv links.csv # also write the table
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from c2p.config import GITHUB_REPO, find_repo_root

URL_RE = re.compile(r"https?://[^\s)>\]]+")
REPO_RE = re.compile(rf"https://github\.com/{re.escape(GITHUB_REPO)}(?:/(blob|tree)/main/(.*))?$")
REFUSED = {401, 403, 429}
ACCEPTED = 202
TIMEOUT_S = 30
# A page that certainly exists, per host that hides every page from scripts behind an error.
HOST_CONTROLS = {"www.kaggle.com": "https://www.kaggle.com/datasets/uciml/iris"}


@dataclass(frozen=True)
class LinkResult:
    """Outcome of one request: ``ok``, ``refused`` (served to browsers only) or ``broken``."""

    url: str
    articles: str
    verdict: str
    status: str
    final_url: str
    title: str


def collect_urls(root: Path) -> dict[str, set[str]]:
    """Every distinct URL of the drafts and their Medium copies, with the parts that cite it."""
    found: dict[str, set[str]] = {}
    for folder in (root / "articles", root / "articles" / "medium"):
        for path in sorted(folder.glob("0*.md")):
            for url in URL_RE.findall(path.read_text(encoding="utf-8")):
                found.setdefault(url.rstrip(".,;:*`"), set()).add(path.name[:2])
    return found


def check_repo_link(url: str, root: Path) -> tuple[str, str] | None:
    """Verdict for a link into this repository, or ``None`` if ``url`` is not one."""
    match = REPO_RE.match(url)
    if match is None:
        return None
    kind, path = match.group(1), match.group(2) or ""
    target = root / path
    exists = target.is_dir() if kind in (None, "tree") else target.is_file()
    return ("ok", "in the working tree") if exists else ("broken", "not in the working tree")


def page_title(body: str) -> str:
    """The ``<title>`` of an HTML page, cleaned up, or an empty string."""
    match = re.search(r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL)
    return " ".join(html.unescape(match.group(1)).split())[:90] if match else ""


def check_doi(url: str) -> tuple[str, str, str]:
    """Ask the DOI resolver where ``url`` points: (verdict, status, target)."""
    response = requests.head(url, timeout=TIMEOUT_S, allow_redirects=False)
    target = response.headers.get("location", "")
    resolves = response.status_code in (301, 302, 303, 307, 308) and bool(target)
    return ("ok" if resolves else "broken", f"{response.status_code} resolves", target)


def same_failure_on_control(url: str, status: int) -> bool:
    """Whether a page of the same host that certainly exists fails with the same status."""
    host = url.split("/")[2]
    control = HOST_CONTROLS.get(host)
    if control is None or control == url:
        return False
    try:
        return requests.get(control, timeout=TIMEOUT_S).status_code == status
    except requests.RequestException:
        return False


def check_url(url: str, articles: set[str], root: Path) -> LinkResult:
    """Request ``url`` once (or look it up in the working tree) and classify the answer."""
    cited = ",".join(sorted(articles))
    local = check_repo_link(url, root)
    if local is not None:
        return LinkResult(url, cited, local[0], local[1], "", "")
    try:
        if url.startswith("https://doi.org/"):
            verdict, status_text, target = check_doi(url)
            return LinkResult(url, cited, verdict, status_text, target, "")
        response = requests.get(url, timeout=TIMEOUT_S, allow_redirects=True)
    except requests.RequestException as error:
        return LinkResult(url, cited, "broken", type(error).__name__, "", str(error)[:90])
    status = response.status_code
    if status == ACCEPTED and not response.text.strip():
        # "Accepted", with no page: what EUR-Lex answers to a script while a browser gets the text.
        return LinkResult(url, cited, "refused", f"{status} with an empty body", "", "")
    verdict = "ok" if status < 400 else "refused" if status in REFUSED else "broken"
    if verdict == "broken" and same_failure_on_control(url, status):
        return LinkResult(url, cited, "refused", f"{status} for any page", "", "")
    final = response.url if response.url != url else ""
    title = page_title(response.text) if "html" in response.headers.get("content-type", "") else ""
    return LinkResult(url, cited, verdict, str(status), final, title)


def check_links(root: Path | None = None, *, workers: int = 8) -> pd.DataFrame:
    """Check every link of the drafts; one row per URL, broken ones first."""
    root = root if root is not None else find_repo_root()
    urls = collect_urls(root)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda item: check_url(item[0], item[1], root), urls.items()))
    order = {"broken": 0, "refused": 1, "ok": 2}
    frame = pd.DataFrame([result.__dict__ for result in results])
    return frame.sort_values("verdict", key=lambda column: column.map(order)).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the table and return 1 if any link is broken."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", default=None, help="also write the table to this file")
    args = parser.parse_args(argv)
    frame = check_links()
    if args.csv:
        frame.to_csv(args.csv, index=False)
    for row in frame.itertuples():
        note = row.title or row.final_url
        print(f"{row.verdict:<8} {row.status:<22} [{row.articles}] {row.url}  {note}")
    counts = frame["verdict"].value_counts().to_dict()
    print(f"{len(frame)} links: {counts}")
    return 1 if counts.get("broken", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
