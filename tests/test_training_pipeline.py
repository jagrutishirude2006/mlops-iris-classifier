
"""
Smoke tests for the model training pipeline.
Checks model training, prediction output, and probabilities.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]


@pytest.fixture
def synthetic_dataset():
    """Create a reproducible synthetic classification dataset."""
    rng = np.random.default_rng(42)
    n_samples = 30

    X = pd.DataFrame(
        rng.normal(5, 1, size=(n_samples, 4)),
        columns=FEATURE_COLS,
    )

    # Ensure all three classes are represented.
    y = np.tile([0, 1, 2], n_samples // 3)

    return X, y


@pytest.fixture
def trained_model(synthetic_dataset):
    """Train and return a Random Forest model."""
    X, y = synthetic_dataset

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)
    return model


def test_model_trains_without_error(trained_model):
    """Check that the model is fitted successfully."""
    assert hasattr(trained_model, "predict")
    assert hasattr(trained_model, "classes_")
    assert len(trained_model.classes_) == 3


def test_model_predict_output_shape(
    synthetic_dataset, trained_model
):
    """Check that predictions have the correct shape."""
    X, _ = synthetic_dataset

    predictions = trained_model.predict(X)

    assert predictions.shape == (len(X),)
    assert set(np.unique(predictions)).issubset({0, 1, 2})


def test_model_predict_proba_sums_to_one(
    synthetic_dataset, trained_model
):
    """Check that prediction probabilities are valid."""
    X, _ = synthetic_dataset

    probabilities = trained_model.predict_proba(X)

    assert probabilities.shape == (len(X), 3)
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    row_sums = probabilities.sum(axis=1)

    assert np.allclose(row_sums, 1.0, atol=1e-6)
