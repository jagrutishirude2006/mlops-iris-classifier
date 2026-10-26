# tests/test_features.py
"""
Unit tests for feature engineering (src/pipeline/features.py).
"""
import pandas as pd

from src.pipeline.features import engineer_features


def test_sepal_and_petal_area_computed_correctly(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [4.0],
        "petal length (cm)": [2.0],
        "petal width (cm)": [1.0],
        "species": ["setosa"],
    })
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    df.to_csv(in_path, index=False)

    result = engineer_features(str(in_path), str(out_path))

    assert result.loc[0, "sepal_area"] == 20.0
    assert result.loc[0, "petal_area"] == 2.0
    assert result.loc[0, "sepal_to_petal_length_ratio"] == 2.5


def test_petal_length_bin_assigns_expected_category(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0, 5.0, 5.0],
        "sepal width (cm)": [3.0, 3.0, 3.0],
        "petal length (cm)": [1.0, 3.0, 6.0],
        "petal width (cm)": [0.2, 1.0, 2.0],
        "species": ["setosa", "versicolor", "virginica"],
    })
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    df.to_csv(in_path, index=False)

    result = engineer_features(str(in_path), str(out_path))

    assert list(result["petal_length_bin"]) == ["short", "medium", "long"]


def test_engineer_features_handles_zero_petal_length_without_crashing(tmp_path):
    df = pd.DataFrame({
        "sepal length (cm)": [5.0],
        "sepal width (cm)": [3.0],
        "petal length (cm)": [0.0],
        "petal width (cm)": [0.2],
        "species": ["setosa"],
    })
    in_path = tmp_path / "in.csv"
    out_path = tmp_path / "out.csv"
    df.to_csv(in_path, index=False)

    result = engineer_features(str(in_path), str(out_path))
    assert pd.isna(result.loc[0, "sepal_to_petal_length_ratio"]), \
        "Division by zero petal length should yield NaN, not a crash or Inf"
