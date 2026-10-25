"""Point the repository links at your GitHub repository.

The README badges, the "Open in Colab" links in the notebooks and ``GITHUB_REPO`` in
``src/c2p/config.py`` use the placeholder ``YOUR_GITHUB_USER/chronos2-in-practice``. Run this
once, before publishing the repository, with its real ``owner/name``::

    uv run python scripts/set_repo_url.py my-user/chronos2-in-practice --dry-run
    uv run python scripts/set_repo_url.py my-user/chronos2-in-practice

Only the placeholder is replaced. Once the real name is in place, running the script again,
with any name, changes nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

PLACEHOLDER = "YOUR_GITHUB_USER/chronos2-in-practice"
# Files that must exist under the repository root, and globs that may match nothing yet.
REQUIRED_FILES: tuple[str, ...] = ("README.md", "src/c2p/config.py")
OPTIONAL_GLOBS: tuple[str, ...] = ("notebooks/*.ipynb",)

# GitHub rules: owners have 1-39 alphanumerics or single inner hyphens; repository names have
# 1-100 letters, digits, ".", "-" or "_" and cannot be "." or "..".
_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}"
_NAME = r"(?!\.{1,2}$)[A-Za-z0-9._-]{1,100}"
REPO_PATTERN = re.compile(rf"(?P<owner>{_OWNER})/(?P<name>{_NAME})")
# A pasted repository URL (https or ssh), possibly followed by a path such as /tree/main.
_GITHUB_URL = re.compile(
    r"(?:https?://(?:www\.)?github\.com/|git@github\.com:)"
    r"(?P<owner>[^/\s]+)/(?P<name>[^/\s?#]+)(?:[/?#].*)?"
)

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def _is_valid(value: str) -> bool:
    # A trailing ".git" comes from a clone URL, not from the repository name.
    return REPO_PATTERN.fullmatch(value) is not None and not value.endswith(".git")


def suggest_repo(value: str) -> str | None:
    """Guess the ``owner/name`` meant by an invalid value, e.g. a pasted clone URL.

    Returns ``None`` unless the guess is itself valid, so a hint never leads to another error.
    """
    text = value.strip()
    if match := _GITHUB_URL.fullmatch(text):
        text = f"{match['owner']}/{match['name']}"
    candidate = text.removesuffix(".git")
    return candidate if _is_valid(candidate) else None


def validate_repo(value: str) -> str:
    """Return ``value`` if it is a valid GitHub ``owner/name``, else raise ``ValueError``."""
    if _is_valid(value):
        return value
    message = (
        f"{value!r} is not a GitHub repository in the form 'owner/name' "
        "(e.g. 'my-user/chronos2-in-practice'; no URL, no '.git', no spaces)."
    )
    if suggestion := suggest_repo(value):
        message += f" Did you mean {suggestion!r}?"
    raise ValueError(message)


def target_files(root: Path) -> list[Path]:
    """List the files that may contain the placeholder, raising if a required one is missing."""
    missing = [rel for rel in REQUIRED_FILES if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(missing)} not found under {root}. Is --root the repository root?"
        )
    files = [root / rel for rel in REQUIRED_FILES]
    for pattern in OPTIONAL_GLOBS:
        files.extend(sorted(path for path in root.glob(pattern) if path.is_file()))
    return files


def replace_in_file(path: Path, new: str, *, dry_run: bool = False) -> int:
    """Replace the placeholder with ``new`` in ``path`` and return the number of occurrences.

    The file is read and written as UTF-8 bytes so line endings are preserved on every OS.
    """
    text = path.read_bytes().decode("utf-8")
    count = text.count(PLACEHOLDER)
    if count and not dry_run:
        path.write_bytes(text.replace(PLACEHOLDER, new).encode("utf-8"))
    return count


def set_repo_url(root: Path, repo: str, *, dry_run: bool = False) -> dict[Path, int]:
    """Replace the placeholder with ``repo`` in every target file under ``root``.

    Returns
    -------
    dict[Path, int]
        Changed (or, with ``dry_run``, to-be-changed) files mapped to the number of
        replacements; empty when the placeholder is no longer present.
    """
    repo = validate_repo(repo)
    changes: dict[Path, int] = {}
    for path in target_files(root):
        if count := replace_in_file(path, repo, dry_run=dry_run):
            changes[path] = count
    return changes


def _repo_arg(value: str) -> str:
    try:
        return validate_repo(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    parser = argparse.ArgumentParser(
        description=f"Replace {PLACEHOLDER} with your GitHub repository in the README, "
        "the notebooks and src/c2p/config.py."
    )
    parser.add_argument("repo", type=_repo_arg, help="GitHub repository as owner/name.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Repository root (default: the folder above scripts/).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would change without writing."
    )
    return parser


def _report(changes: dict[Path, int], root: Path, repo: str, *, dry_run: bool) -> str:
    if not changes:
        return f"No file under {root} contains {PLACEHOLDER}; nothing to do."
    verb = "Would replace" if dry_run else "Replaced"
    lines = [f"{verb} {PLACEHOLDER} -> {repo} in {len(changes)} file(s):"]
    lines += [f"  {path.relative_to(root).as_posix()} ({n})" for path, n in changes.items()]
    if dry_run:
        lines.append("Dry run: no file was written.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root {root} is not a directory.")
    try:
        changes = set_repo_url(root, args.repo, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    print(_report(changes, root, args.repo, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
