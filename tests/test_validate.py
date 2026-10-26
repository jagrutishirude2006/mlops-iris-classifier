# tests/test_validate.py
"""
Unit tests for the validation stage (src/pipeline/validate.py),
confirming both PASS and FAIL conditions are correctly detected.
"""
import pandas as pd
import pytest

from src.pipeline.validate import validate, DataValidationError

VALID_ROW = {
    "sepal length (cm)": 5.1, "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4, "petal width (cm)": 0.2,
    "species": "setosa", "sepal_area": 17.85, "petal_area": 0.28,
    "sepal_to_petal_length_ratio": 3.64, "petal_length_bin": "short",
}

def _write(tmp_path, df):
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_validate_passes_on_clean_data(tmp_path):
    df = pd.DataFrame([VALID_ROW, VALID_ROW])
    path = _write(tmp_path, df)
    result = validate(path)
    assert len(result) == 2


def test_validate_fails_on_missing_column(tmp_path):
    row = VALID_ROW.copy()
    df = pd.DataFrame([row]).drop(columns=["sepal_area"])
    path = _write(tmp_path, df)
    with pytest.raises(DataValidationError, match="Missing expected columns"):
        validate(path)


def test_validate_fails_on_invalid_species(tmp_path):
    row = VALID_ROW.copy()
    row["species"] = "unicorn"
    df = pd.DataFrame([row])
    path = _write(tmp_path, df)
    with pytest.raises(DataValidationError, match="Unexpected species values"):
        validate(path)
        
def test_validate_fails_on_out_of_range_value(tmp_path):
    row = VALID_ROW.copy()
    row["sepal length (cm)"] = 50.0  # far outside the 3.0-9.0 expected range
    df = pd.DataFrame([row])
    path = _write(tmp_path, df)
    with pytest.raises(DataValidationError, match="out of expected range"):
        validate(path)
