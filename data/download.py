"""Download the Spanish energy dataset used throughout the series.

Run from the repository root::

    uv run python data/download.py                  # into data/raw/spain_energy/
    uv run python data/download.py --force          # download and copy again
    uv run python data/download.py --dest some/dir  # somewhere else

The files come from the CC0 Kaggle dataset
``nicholasjhana/energy-consumption-generation-prices-and-weather`` (no Kaggle account needed)
and are checked against known SHA256 hashes. They are never committed: ``data/raw/`` is in
``.gitignore``. Notebooks call :func:`c2p.data.download_spain_energy` themselves, so running
this script first is optional.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from c2p.data import SPAIN_ENERGY_FILES, download_spain_energy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (``argv=None`` reads ``sys.argv``)."""
    parser = argparse.ArgumentParser(description="Download the Spanish energy dataset (CC0).")
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="target folder (default: data/raw/spain_energy in the repository)",
    )
    parser.add_argument(
        "--force", action="store_true", help="download again even if the files exist"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Download the dataset and print where the files are."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    folder = download_spain_energy(args.dest, force=args.force)
    print(f"Spanish energy dataset ready in {folder.resolve()}")
    for name in SPAIN_ENERGY_FILES:
        size_mb = (folder / name).stat().st_size / 1e6
        print(f"  {name} ({size_mb:.1f} MB, SHA256 verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
