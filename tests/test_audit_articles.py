"""The article drafts must agree with ``results/`` and with the package they show.

``scripts/audit_articles.py`` rebuilds every numeric table row from the CSV files and binds
every ``c2p`` call of the code blocks to its real signature. Running it in the fast suite means
an edit to an article, a re-run that moves a result, or a renamed argument cannot land without
the three staying in step.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_articles.py"
spec = importlib.util.spec_from_file_location("audit_articles", SCRIPT)
audit = importlib.util.module_from_spec(spec)
sys.modules["audit_articles"] = audit
spec.loader.exec_module(audit)


def test_articles_agree_with_results_and_package() -> None:
    assert audit.audit() == []


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (0.018492 * 100, 1, "1.8"),  # not 1.9: rounded once, from the stored value
        (0.7625 * 100, 1, "76.3"),  # a true tie rounds half up
        (76.24999999999999, 1, "76.3"),  # binary noise around a tie is snapped first
        (-0.248474 * 100, 1, "24.8"),
        (810.95, 0, "811"),
    ],
)
def test_half_up_rounds_once_from_the_stored_value(
    value: float, decimals: int, expected: str
) -> None:
    assert audit.half_up(value, decimals) == expected


def test_signed_and_percent_formats_use_a_true_minus() -> None:
    assert audit.spct(0.018492) == "+1.8"
    assert audit.spct(-0.0082) == "\u22120.8"
    assert audit.negative(-0.00007, 5) == "\u22120.00007"
    assert audit.thousands(4210.9) == "4,211"
    assert audit.thousands(1234.75, 1) == "1,234.8"


def test_check_rows_reports_a_row_typed_from_a_rounded_printout() -> None:
    results = audit.Results(SCRIPT.parents[1])
    (path,) = sorted((SCRIPT.parents[1] / "articles").glob("04-*.md"))
    text = path.read_text(encoding="utf-8")
    assert audit.check_rows("04", text, results) == []
    match = re.search(r"\+(\d)\.\d% \[[^\]]+\]", text)
    assert match is not None
    right = match.group(0)
    wrong = text.replace(right, f"+{(int(match.group(1)) + 1) % 10}{right[2:]}")
    problems = audit.check_rows("04", wrong, results)
    assert problems and right in problems[0]


def test_check_code_binds_calls_across_blocks() -> None:
    good = (
        "```python\nfrom c2p.finetune import lora_config\n```\n"
        "text\n```python\nconfig = lora_config(r=8)\n```\n"
    )
    assert audit.check_code("xx", good) == []
    bad = good.replace("lora_config(r=8)", "lora_config(rank=8)")
    (problem,) = audit.check_code("xx", bad)
    assert "lora_config" in problem and "rank" in problem
    missing = "```python\nfrom c2p.finetune import no_such_name\n```\n"
    assert audit.check_code("xx", missing) == ["xx: c2p.finetune has no 'no_such_name'"]


def test_check_text_flags_em_dashes_and_rewrite_leftovers() -> None:
    assert audit.check_text("xx", "A clean paragraph.") == []
    problems = audit.check_text("xx", "one \u2014 two\nSure, here it goes")
    assert len(problems) == 2
    assert "em dash" in problems[0] and "rewrite leftover" in problems[1]
