from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

RAW_PATH = REPO_ROOT / "datasets" / "census_income.csv"


def _ensure_clean_dataset() -> Path:
    """Create the cleaned dataset when it is missing and return its path."""
    from task2.data import DATA_PATH_ENV, DEFAULT_DATA_PATH

    data_path = Path(os.environ.get(DATA_PATH_ENV, DEFAULT_DATA_PATH))
    if data_path.exists():
        return data_path

    from task2.cleaning import clean_census, cleaning_summary

    raw = pd.read_csv(RAW_PATH, encoding="utf-8")
    cleaned = clean_census(raw)

    summary = {
        "source": RAW_PATH.relative_to(REPO_ROOT).as_posix(),
        **cleaning_summary(raw, cleaned),
    }

    data_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(data_path, index=False)

    meta_path = data_path.with_name(f"{data_path.stem}_meta.json")
    meta_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        "Cleaned dataset not found; generated it automatically.\n"
        f"  Raw rows: {summary['rows_before']:,}\n"
        f"  Rows removed (sentinel values): "
        f"{summary['rows_removed_sentinel']:,}\n"
        f"  Cleaned rows: {summary['rows_after']:,}\n"
        f"  Positive class: {summary['positive_count']:,} "
        f"({summary['positive_rate'] * 100:.2f}%)\n"
        f"  Saved: {data_path}\n"
        f"  Saved metadata: {meta_path}"
    )

    return data_path


def main() -> int:
    _ensure_clean_dataset()

    import task2

    task2.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
