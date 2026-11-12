from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from task2.cleaning import (
    CATEGORICAL_FEATURES,
    CLEAN_COLUMN_ORDER,
    clean_census,
    cleaning_summary,
    categorical_columns,
    numeric_columns,
)

RAW_PATH = (
    Path(__file__).resolve().parents[2] / "datasets" / "census_income.csv"
)


def make_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            " age": [30, 40, 50],
            " workclass": [" ?", " Private", " Private"],
            " fnlwgt": [100, 200, 300],
            " education": [" Bachelors", " HS-grad", " Masters"],
            " education-num": [13, 9, 14],
            " marital-status": [
                " Never-married",
                " Married-civ-spouse",
                " Divorced",
            ],
            " occupation": [" Adm-clerical", " Exec-managerial", " ?"],
            " relationship": [" Not-in-family", " Husband", " Unmarried"],
            " race": [" White", " Black", " White"],
            " sex": [" Male", " Female", " Male"],
            " capital-gain": [0, 99999, 5000],
            " capital-loss": [0, 0, 0],
            " hours-per-week": [40, 50, 60],
            " native-country": [" United-States", " Mexico", " ?"],
            " income": [" <=50K.", " >50K.", " >50K"],
        }
    )


class CleaningTests(unittest.TestCase):
    def test_clean_census_rules(self) -> None:
        raw = make_raw_frame()
        raw_shape = raw.shape
        raw_columns = list(raw.columns)

        cleaned = clean_census(raw)

        self.assertEqual(cleaned.shape, (2, len(CLEAN_COLUMN_ORDER)))
        self.assertEqual(list(cleaned.columns), list(CLEAN_COLUMN_ORDER))
        self.assertNotIn("fnlwgt", cleaned.columns)
        self.assertNotIn("education-num", cleaned.columns)
        self.assertTrue(pd.api.types.is_bool_dtype(cleaned["target"]))
        self.assertEqual(cleaned["target"].tolist(), [False, True])
        self.assertEqual(cleaned["income"].tolist(), ["<=50K", ">50K"])
        self.assertEqual(cleaned["workclass"].tolist(), ["Unknown", "Private"])
        self.assertEqual(
            cleaned["occupation"].tolist(),
            ["Adm-clerical", "Unknown"],
        )
        self.assertEqual(
            cleaned["native-country"].tolist(),
            ["United-States", "Unknown"],
        )

        # The input must not be mutated.
        self.assertEqual(raw.shape, raw_shape)
        self.assertEqual(list(raw.columns), raw_columns)
        self.assertEqual(raw[" capital-gain"].tolist(), [0, 99999, 5000])

        # No one-hot encoding is allowed in the output.
        for prefix in (
            "workclass_",
            "education_",
            "marital-status_",
            "occupation_",
            "relationship_",
            "race_",
            "sex_",
            "native-country_",
        ):
            self.assertFalse(
                any(column.startswith(prefix) for column in cleaned.columns),
                msg=f"Unexpected one-hot column with prefix {prefix}",
            )

    def test_clean_census_returns_expected_dtypes(self) -> None:
        cleaned = clean_census(make_raw_frame())

        for column in CATEGORICAL_FEATURES:
            self.assertTrue(
                pd.api.types.is_string_dtype(cleaned[column]),
                msg=f"{column} should be a string column",
            )
        for column in ("age", "capital-gain", "capital-loss", "hours-per-week"):
            self.assertTrue(
                pd.api.types.is_numeric_dtype(cleaned[column]),
                msg=f"{column} should be numeric",
            )

    def test_cleaning_summary(self) -> None:
        raw = make_raw_frame()
        cleaned = clean_census(raw)

        summary = cleaning_summary(raw, cleaned)

        self.assertEqual(summary["rows_before"], 3)
        self.assertEqual(summary["rows_after"], 2)
        self.assertEqual(summary["rows_removed"], 1)
        self.assertEqual(summary["rows_removed_sentinel"], 1)
        self.assertEqual(
            summary["sentinel_counts"],
            {"capital-gain": 1, "capital-loss": 0},
        )
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["positive_rate"], 0.5)
        self.assertEqual(
            summary["missing_markers_raw"],
            {"workclass": 1, "occupation": 1, "native-country": 1},
        )
        self.assertEqual(
            summary["missing_markers_cleaned"],
            {"workclass": 1, "occupation": 1, "native-country": 1},
        )
        self.assertEqual(summary["columns"], list(CLEAN_COLUMN_ORDER))
        self.assertEqual(
            summary["numeric_columns"],
            ["age", "capital-gain", "capital-loss", "hours-per-week"],
        )
        self.assertEqual(
            summary["categorical_columns"],
            list(CATEGORICAL_FEATURES),
        )
        self.assertEqual(summary["target_columns"], ["income", "target"])

    def test_numeric_and_categorical_helpers(self) -> None:
        cleaned = clean_census(make_raw_frame())

        self.assertEqual(
            numeric_columns(cleaned),
            ["age", "capital-gain", "capital-loss", "hours-per-week"],
        )
        self.assertEqual(
            categorical_columns(cleaned),
            list(CATEGORICAL_FEATURES),
        )
        self.assertNotIn("target", categorical_columns(cleaned))
        self.assertNotIn("income", categorical_columns(cleaned))

    def test_missing_required_column_raises(self) -> None:
        raw = make_raw_frame().drop(columns=[" capital-gain"])

        with self.assertRaises(KeyError):
            clean_census(raw)

    def test_unexpected_income_value_raises(self) -> None:
        raw = make_raw_frame()
        raw.loc[0, " income"] = " unknown"

        with self.assertRaises(ValueError):
            clean_census(raw)


class RealDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = pd.read_csv(RAW_PATH)
        cls.cleaned = clean_census(cls.raw)
        cls.summary = cleaning_summary(cls.raw, cls.cleaned)

    def test_shape_and_columns(self) -> None:
        self.assertEqual(self.cleaned.shape, (48_598, 14))
        self.assertEqual(list(self.cleaned.columns), list(CLEAN_COLUMN_ORDER))
        self.assertNotIn("fnlwgt", self.cleaned.columns)
        self.assertNotIn("education-num", self.cleaned.columns)
        self.assertTrue(pd.api.types.is_bool_dtype(self.cleaned["target"]))

    def test_target_and_sentinel_values(self) -> None:
        self.assertEqual(int(self.cleaned["target"].sum()), 11_443)
        self.assertAlmostEqual(
            float(self.cleaned["target"].mean()),
            0.2354623647,
            places=8,
        )
        self.assertEqual(int(self.cleaned["capital-gain"].max()), 41_310)
        self.assertLess(int(self.cleaned["capital-gain"].max()), 99_999)

    def test_question_marks_are_replaced(self) -> None:
        expected = {
            "workclass": 2_794,
            "occupation": 2_804,
            "native-country": 847,
        }
        for column, count in expected.items():
            self.assertEqual(
                int(self.cleaned[column].eq("Unknown").sum()),
                count,
                msg=f"Unexpected Unknown count in {column}",
            )
            self.assertFalse(
                self.cleaned[column].eq("?").any(),
                msg=f"Unreplaced ? in {column}",
            )

    def test_summary_matches_real_dataset(self) -> None:
        self.assertEqual(self.summary["rows_before"], 48_842)
        self.assertEqual(self.summary["rows_after"], 48_598)
        self.assertEqual(self.summary["rows_removed"], 244)
        self.assertEqual(self.summary["rows_removed_sentinel"], 244)
        self.assertEqual(
            self.summary["sentinel_counts"],
            {"capital-gain": 244, "capital-loss": 0},
        )
        self.assertEqual(self.summary["positive_count"], 11_443)
        self.assertEqual(self.summary["negative_count"], 37_155)
        self.assertAlmostEqual(self.summary["positive_rate"], 0.2354623647, places=8)
        self.assertEqual(
            self.summary["numeric_columns"],
            ["age", "capital-gain", "capital-loss", "hours-per-week"],
        )
        self.assertEqual(
            self.summary["categorical_columns"],
            list(CATEGORICAL_FEATURES),
        )

    def test_no_one_hot_columns(self) -> None:
        for prefix in (
            "workclass_",
            "education_",
            "marital-status_",
            "occupation_",
            "relationship_",
            "race_",
            "sex_",
            "native-country_",
        ):
            self.assertFalse(
                any(column.startswith(prefix) for column in self.cleaned.columns),
                msg=f"Unexpected one-hot column with prefix {prefix}",
            )


if __name__ == "__main__":
    unittest.main()
