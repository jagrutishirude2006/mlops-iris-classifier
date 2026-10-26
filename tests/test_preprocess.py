
"""
Unit tests for the preprocessing stage (src/pipeline/preprocess.py).
Tests normal operation and edge cases.
"""

import pandas as pd
import pytest

from src.pipeline.preprocess import preprocess


NUMERIC_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def valid_raw_df(tmp_path):
    """Create a sample raw Iris dataset."""
    df = pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7],
        "sepal width (cm)": [3.5, 3.0, 3.2],
        "petal length (cm)": [1.4, 1.4, 1.3],
        "petal width (cm)": [0.2, 0.2, 0.2],
        "species": ["setosa", "setosa", "setosa"],
        "collected_at": ["2026-01-01"] * 3,
    })

    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_preprocess_removes_duplicates(tmp_path, valid_raw_df):
    df = pd.read_csv(valid_raw_df)

    # Add one duplicate row
    df_with_dupe = pd.concat(
        [df, df.iloc[[0]]],
        ignore_index=True,
    )

    input_path = tmp_path / "raw_with_dupe.csv"
    output_path = tmp_path / "out.csv"
    df_with_dupe.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 3, "Duplicate row was not removed"
    assert output_path.exists(), "Output CSV was not created"


def test_preprocess_imputes_missing_numeric_values(
    tmp_path, valid_raw_df
):
    df = pd.read_csv(valid_raw_df)

    # Introduce a missing numeric value
    df.loc[0, "sepal length (cm)"] = float("nan")

    input_path = tmp_path / "raw_with_missing.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    expected_median = df["sepal length (cm)"].median()

    result = preprocess(str(input_path), str(output_path))

    assert result["sepal length (cm)"].isna().sum() == 0, (
        "Missing numeric value was not imputed"
    )

    assert result.loc[0, "sepal length (cm)"] == pytest.approx(
        expected_median
    )


def test_preprocess_drops_rows_with_missing_target(
    tmp_path, valid_raw_df
):
    df = pd.read_csv(valid_raw_df)

    # Remove one target label
    df.loc[0, "species"] = None

    input_path = tmp_path / "raw_missing_target.csv"
    output_path = tmp_path / "out.csv"
    df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 2, (
        "Row with missing target label was not dropped"
    )

    assert result["species"].notna().all()


def test_preprocess_handles_empty_dataframe(tmp_path):
    empty_df = pd.DataFrame(
        columns=NUMERIC_COLS + ["species", "collected_at"]
    )

    input_path = tmp_path / "empty.csv"
    output_path = tmp_path / "out.csv"
    empty_df.to_csv(input_path, index=False)

    result = preprocess(str(input_path), str(output_path))

    assert len(result) == 0
    assert list(result.columns[:4]) == NUMERIC_COLS
    assert output_path.exists(), "Output CSV was not created"
