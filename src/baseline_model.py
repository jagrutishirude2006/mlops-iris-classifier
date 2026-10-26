# src/baseline_model.py
"""
Establishes a simple baseline model (default-hyperparameter Decision Tree)
against which all subsequent tuning efforts in this experiment are measured.
"""
import mlflow
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
FEATURE_COLS = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
    "sepal_area", "petal_area", "sepal_to_petal_length_ratio",
]


def load_dataset(path: str):
    df = pd.read_csv(path)
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_baseline(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    with mlflow.start_run(run_name="baseline_decision_tree"):
        model = DecisionTreeClassifier(random_state=42)  # all defaults
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")

        mlflow.log_param("model_type", "DecisionTreeClassifier_default")
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std", cv_scores.std())

        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_score)

        print(f"Baseline CV f1_macro: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"Baseline test accuracy: {test_score:.4f}")
        return cv_scores.mean()


if __name__ == "__main__":
    run_baseline("data/processed/iris_features.csv")
