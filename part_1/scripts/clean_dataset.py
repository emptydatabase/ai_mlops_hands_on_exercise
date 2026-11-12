from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from task2.cleaning import clean_census, cleaning_summary

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent

RAW_PATH = REPO_ROOT / "datasets" / "census_income.csv"
OUTPUT_DIR = ROOT / "dataset"
OUTPUT_PATH = OUTPUT_DIR / "census_clean.csv"
METADATA_PATH = OUTPUT_DIR / "census_clean_meta.json"


def main() -> int:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8")
    cleaned = clean_census(raw)

    summary = cleaning_summary(raw, cleaned)
    summary = {
        "source": RAW_PATH.relative_to(REPO_ROOT).as_posix(),
        **summary,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    METADATA_PATH.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Raw rows: {summary['rows_before']:,}")
    print(
        "Rows removed (sentinel values): "
        f"{summary['rows_removed_sentinel']:,}"
    )
    print(f"Cleaned rows: {summary['rows_after']:,}")
    print(
        "Positive class: "
        f"{summary['positive_count']:,} "
        f"({summary['positive_rate'] * 100:.2f}%)"
    )
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
