"""The boundary of the companion package: what each module exports, and that callers respect it.

Every module of ``c2p`` declares ``__all__``. Helpers that modules share with one another stay
importable but are left out of it, so the list is what a notebook, a script or an article may
import. These tests keep the list true: every name in it exists, and nothing outside ``src``
imports a name that is not in it.
"""

from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import re
from pathlib import Path

import pytest

import c2p

REPO = Path(__file__).resolve().parents[1]
MODULES = sorted(
    info.name for info in pkgutil.walk_packages(c2p.__path__, prefix="c2p.") if not info.ispkg
)
CALLERS = ("notebooks", "scripts", "bench", "monitoring", "examples", "deploy")


@pytest.mark.parametrize("name", MODULES)
def test_every_module_declares_what_it_exports(name: str) -> None:
    module = importlib.import_module(name)
    exported = getattr(module, "__all__", None)
    assert exported, f"{name} has no __all__"
    assert len(set(exported)) == len(exported), f"{name}.__all__ repeats a name"
    missing = [item for item in exported if not hasattr(module, item)]
    assert not missing, f"{name}.__all__ lists names the module does not define: {missing}"


def code_of(path: Path) -> str:
    """The Python source of a script, a notebook's code cells, or an article's code blocks."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        cells = json.loads(text)["cells"]
        lines = [
            line
            for cell in cells
            if cell["cell_type"] == "code"
            for line in "".join(cell["source"]).splitlines()
            if not line.lstrip().startswith(("!", "%"))
        ]
        return "\n".join(lines)
    if path.suffix == ".md":
        return "\n".join(re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL))
    return text


def imports_from_c2p(source: str) -> list[tuple[str, str]]:
    """Every ``(module, name)`` of the ``from c2p.x import name`` statements of ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # an article block that is a fragment, not a module
        one_line = re.findall(r"^from (c2p[\w.]*) import ([\w, ]+)$", source, flags=re.MULTILINE)
        bracketed = re.findall(r"^from (c2p[\w.]*) import \(([^)]*)\)", source, flags=re.MULTILINE)
        return [
            (module, name.strip())
            for module, names in one_line + bracketed
            for name in names.split(",")
            if name.strip()
        ]
    return [
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("c2p")
        for alias in node.names
    ]


def caller_files() -> list[Path]:
    files = [path for folder in CALLERS for path in (REPO / folder).rglob("*.py")]
    files += sorted((REPO / "notebooks").glob("*.ipynb"))
    files += sorted((REPO / "articles").glob("0*.md"))
    return [path for path in files if "__pycache__" not in path.parts]


def test_callers_import_only_exported_names() -> None:
    outside = []
    for path in caller_files():
        for module, name in imports_from_c2p(code_of(path)):
            exported = getattr(importlib.import_module(module), "__all__", ())
            if name not in exported and f"{module}.{name}" not in MODULES:
                outside.append(f"{path.relative_to(REPO)}: from {module} import {name}")
    assert not outside, "imports of names outside __all__:\n" + "\n".join(outside)


def test_the_import_scanner_reads_modules_notebook_cells_and_fragments() -> None:
    assert imports_from_c2p("from c2p.eval import backtest, evaluate\n") == [
        ("c2p.eval", "backtest"),
        ("c2p.eval", "evaluate"),
    ]
    fragment = "from c2p.models import load_chronos2\nresult = ...(\n"
    assert imports_from_c2p(fragment) == [("c2p.models", "load_chronos2")]
    assert len(caller_files()) > 30
