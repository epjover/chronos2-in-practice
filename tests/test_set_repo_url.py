"""Tests for ``scripts/set_repo_url.py`` (run on temporary copies, never on the real files)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "set_repo_url.py"
NEW_REPO = "jane-doe/chronos2-in-practice"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("set_repo_url", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sru = _load_script()
PLACEHOLDER: str = sru.PLACEHOLDER


def _notebook(placeholder: str) -> str:
    """A minimal notebook whose first cell holds an "Open in Colab" badge."""
    badge = (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/{placeholder}/blob/main/notebooks/00.ipynb)"
    )
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [badge]},
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["print(1)"],
        },
    ]
    notebook = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(notebook, indent=1) + "\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake repository with the placeholder in every kind of target file and elsewhere."""
    files = {
        "README.md": f"# Title\n\nhttps://github.com/{PLACEHOLDER}\n\n`{PLACEHOLDER}`\n",
        "src/c2p/config.py": f'GITHUB_REPO = "{PLACEHOLDER}"\n',
        "notebooks/00_data_checks.ipynb": _notebook(PLACEHOLDER),
        "notebooks/01_quickstart.ipynb": _notebook(PLACEHOLDER),
        "notebooks/scratch.txt": f"{PLACEHOLDER}\n",
        "notebooks/archive/old.ipynb": _notebook(PLACEHOLDER),
        "src/c2p/data.py": f"# {PLACEHOLDER}\n",
        "docs/notes.md": f"{PLACEHOLDER}\n",
    }
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return tmp_path


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "value",
    [
        "jane-doe/chronos2-in-practice",
        "a/b",
        "Org42/repo.name_v2",
        "user/.github",
        f"{'a' * 39}/{'b' * 100}",
    ],
)
def test_validate_repo_accepts_github_names(value: str) -> None:
    assert sru.validate_repo(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "no-slash",
        "a/b/c",
        "/repo",
        "user/",
        "-user/repo",
        "user-/repo",
        "us--er/repo",
        "user/re po",
        "user/.",
        "user/..",
        "user/repo.git",
        "https://github.com/user/repo",
        PLACEHOLDER,  # underscores are not allowed in GitHub user names
        f"{'a' * 40}/repo",
        f"user/{'b' * 101}",
    ],
)
def test_validate_repo_rejects_invalid_names(value: str) -> None:
    with pytest.raises(ValueError, match="owner/name"):
        sru.validate_repo(value)


@pytest.mark.parametrize(
    ("value", "suggestion"),
    [
        ("user/repo.git", "user/repo"),
        ("https://github.com/jane/chronos2-in-practice.git", "jane/chronos2-in-practice"),
        ("https://github.com/jane/chronos2-in-practice", "jane/chronos2-in-practice"),
        ("https://github.com/jane/chronos2-in-practice/", "jane/chronos2-in-practice"),
        ("http://www.github.com/jane/repo/tree/main", "jane/repo"),
        ("git@github.com:jane/repo.git", "jane/repo"),
        (" jane/repo ", "jane/repo"),
    ],
)
def test_invalid_repo_error_suggests_the_owner_name(value: str, suggestion: str) -> None:
    with pytest.raises(ValueError, match="owner/name") as exc_info:
        sru.validate_repo(value)

    assert f"Did you mean {suggestion!r}?" in str(exc_info.value)
    assert sru.validate_repo(suggestion) == suggestion


@pytest.mark.parametrize(
    "value",
    [
        "user/.git",
        "https://github.com/user/.git",
        "user/",
        "not a repo",
        "https://gitlab.com/user/repo",
        PLACEHOLDER,
    ],
)
def test_invalid_repo_error_never_suggests_an_invalid_name(value: str) -> None:
    with pytest.raises(ValueError, match="owner/name") as exc_info:
        sru.validate_repo(value)

    assert "Did you mean" not in str(exc_info.value)


def test_replaces_placeholder_only_in_target_files(repo: Path) -> None:
    before = _snapshot(repo)

    changes = sru.set_repo_url(repo, NEW_REPO)

    changed = {path.relative_to(repo).as_posix(): n for path, n in changes.items()}
    assert changed == {
        "README.md": 2,
        "src/c2p/config.py": 1,
        "notebooks/00_data_checks.ipynb": 1,
        "notebooks/01_quickstart.ipynb": 1,
    }
    after = _snapshot(repo)
    for rel, content in before.items():
        expected = content.replace(PLACEHOLDER.encode(), NEW_REPO.encode())
        assert after[rel] == (expected if rel in changed else content), rel


def test_notebooks_stay_valid_json_with_updated_colab_link(repo: Path) -> None:
    sru.set_repo_url(repo, NEW_REPO)

    notebook = json.loads((repo / "notebooks/00_data_checks.ipynb").read_text("utf-8"))
    badge = "".join(notebook["cells"][0]["source"])
    assert f"colab.research.google.com/github/{NEW_REPO}/blob/main/" in badge
    assert PLACEHOLDER not in badge


def test_config_gets_new_repo(repo: Path) -> None:
    sru.set_repo_url(repo, NEW_REPO)

    assert (repo / "src/c2p/config.py").read_text("utf-8") == f'GITHUB_REPO = "{NEW_REPO}"\n'


def test_dry_run_reports_without_writing(repo: Path) -> None:
    before = _snapshot(repo)

    changes = sru.set_repo_url(repo, NEW_REPO, dry_run=True)

    assert len(changes) == 4
    assert _snapshot(repo) == before


def test_second_run_is_a_no_op(repo: Path) -> None:
    sru.set_repo_url(repo, NEW_REPO)
    after_first = _snapshot(repo)

    assert sru.set_repo_url(repo, "someone-else/fork") == {}
    assert _snapshot(repo) == after_first


def test_line_endings_are_preserved(repo: Path) -> None:
    readme = repo / "README.md"
    readme.write_bytes(f"# Title\r\n{PLACEHOLDER}\r\n".encode())

    sru.set_repo_url(repo, NEW_REPO)

    assert readme.read_bytes() == f"# Title\r\n{NEW_REPO}\r\n".encode()


def test_works_without_notebooks_folder(repo: Path) -> None:
    shutil.rmtree(repo / "notebooks")

    changes = sru.set_repo_url(repo, NEW_REPO)

    assert {path.name for path in changes} == {"README.md", "config.py"}


@pytest.mark.parametrize("missing", ["README.md", "src/c2p/config.py"])
def test_missing_required_file_raises(repo: Path, missing: str) -> None:
    (repo / missing).unlink()

    with pytest.raises(FileNotFoundError, match="--root"):
        sru.set_repo_url(repo, NEW_REPO)


def test_invalid_repo_is_rejected_before_touching_files(repo: Path) -> None:
    before = _snapshot(repo)

    with pytest.raises(ValueError, match="owner/name"):
        sru.set_repo_url(repo, "not a repo")
    assert _snapshot(repo) == before


def test_real_files_copied_to_tmp(tmp_path: Path) -> None:
    """The real README and config are valid targets; replacing in copies leaves no placeholder."""
    originals: dict[str, bytes] = {}
    for rel in sru.REQUIRED_FILES:
        source = REPO_ROOT / rel
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        originals[rel] = source.read_bytes()

    sru.set_repo_url(tmp_path, NEW_REPO)

    for rel, content in originals.items():
        updated = (tmp_path / rel).read_bytes()
        assert updated == content.replace(PLACEHOLDER.encode(), NEW_REPO.encode())
        assert PLACEHOLDER.encode() not in updated


def _clone_folders_and_cd_targets(markdown: str) -> list[tuple[str, str]]:
    """Pair the folder each ``git clone`` creates with the ``cd``/``%cd`` on the next line."""
    lines = markdown.splitlines()
    pairs: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        if "git clone" not in line:
            continue
        args = line.split()
        url_index = next(i for i, arg in enumerate(args) if "://" in arg)
        url = args[url_index]
        default_folder = url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        folder = args[url_index + 1] if url_index + 1 < len(args) else default_folder
        command, target = lines[index + 1].split()[:2]
        assert command in {"cd", "%cd"}, f"no cd after: {line.strip()}"
        pairs.append((folder, target))
    return pairs


def test_readme_cd_matches_clone_folder_for_any_repo_name(tmp_path: Path) -> None:
    """Renaming the repository must not break the clone-then-cd setup steps in the README."""
    for rel in sru.REQUIRED_FILES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / rel, target)

    sru.set_repo_url(tmp_path, "jane-doe/my-forecasting-notes")

    pairs = _clone_folders_and_cd_targets((tmp_path / "README.md").read_text("utf-8"))
    assert pairs, "the README has no git clone command"
    for folder, cd_target in pairs:
        assert cd_target == folder


def test_main_prints_changed_files(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert sru.main([NEW_REPO, "--root", str(repo)]) == 0

    out = capsys.readouterr().out
    assert f"Replaced {PLACEHOLDER} -> {NEW_REPO} in 4 file(s):" in out
    for rel in ("README.md (2)", "src/c2p/config.py (1)", "notebooks/00_data_checks.ipynb (1)"):
        assert rel in out
    assert "docs/notes.md" not in out


def test_main_dry_run_says_nothing_was_written(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _snapshot(repo)

    assert sru.main([NEW_REPO, "--root", str(repo), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert out.startswith("Would replace")
    assert "Dry run: no file was written." in out
    assert _snapshot(repo) == before


def test_main_reports_nothing_to_do(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sru.set_repo_url(repo, NEW_REPO)

    assert sru.main([NEW_REPO, "--root", str(repo)]) == 0
    assert "nothing to do" in capsys.readouterr().out


def _usage_error(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Run ``main`` expecting an argparse usage error (exit code 2); return stderr."""
    with pytest.raises(SystemExit) as exc_info:
        sru.main(argv)
    assert exc_info.value.code == 2
    return capsys.readouterr().err


def test_main_rejects_invalid_repo(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = _snapshot(repo)

    err = _usage_error(["not/a/repo", "--root", str(repo)], capsys)

    assert "owner/name" in err
    assert _snapshot(repo) == before


def test_main_suggests_owner_name_for_a_pasted_clone_url(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    clone_url = f"https://github.com/{NEW_REPO}.git"

    err = _usage_error([clone_url, "--root", str(repo)], capsys)

    assert f"Did you mean {NEW_REPO!r}?" in err


def test_main_rejects_missing_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    err = _usage_error([NEW_REPO, "--root", str(tmp_path / "missing")], capsys)

    assert "is not a directory" in err


def test_main_rejects_root_without_repository_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    err = _usage_error([NEW_REPO, "--root", str(tmp_path)], capsys)

    assert "Is --root the repository root?" in err


def test_command_line_entry_point(repo: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), NEW_REPO, "--root", str(repo), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Would replace" in result.stdout
    assert PLACEHOLDER in (repo / "README.md").read_text("utf-8")
