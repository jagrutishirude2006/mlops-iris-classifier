# tests/test_model_quality.py
"""
Model quality gate: trains on the real Experiment 4 feature set and
asserts accuracy exceeds a defined minimum threshold. Intended to be
run in CI to block merges that regress model quality.
"""
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MIN_ACCEPTABLE_ACCURACY = 0.90
FEATURE_COLS = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
    "sepal_area", "petal_area", "sepal_to_petal_length_ratio",
]


@pytest.fixture(scope="module")
def trained_model_and_test_set():
    df = pd.read_csv("data/processed/iris_features.csv")
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test, le


def test_predictions_are_within_known_label_set(trained_model_and_test_set):
    model, X_test, y_test, le = trained_model_and_test_set
    preds = model.predict(X_test)
    assert set(preds).issubset(set(range(len(le.classes_))))


def test_model_meets_minimum_accuracy_threshold(trained_model_and_test_set):
    model, X_test, y_test, le = trained_model_and_test_set
    accuracy = model.score(X_test, y_test)
    assert accuracy >= MIN_ACCEPTABLE_ACCURACY, (
        f"Model accuracy {accuracy:.4f} fell below the minimum acceptable "
        f"threshold of {MIN_ACCEPTABLE_ACCURACY}"
    )
