# src/grid_search_tuning.py
"""
Exhaustive Grid Search over a Random Forest hyperparameter grid,
with every candidate's cross-validated score logged to MLflow.
"""
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS = [
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
    "sepal_area", "petal_area", "sepal_to_petal_length_ratio",
]
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}


def load_dataset(path: str):
    df = pd.read_csv(path)
    le = LabelEncoder()
    y = le.fit_transform(df["species"])
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def run_grid_search(data_path: str):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-hyperparameter-tuning")

    X_train, X_test, y_train, y_test = load_dataset(data_path)

    with mlflow.start_run(run_name="grid_search_random_forest"):
        total_combinations = 1
        for v in PARAM_GRID.values():
            total_combinations *= len(v)
        mlflow.log_param("search_type", "GridSearchCV")
        mlflow.log_param("total_combinations", total_combinations)
        mlflow.log_param("cv_folds", 5)

        grid = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid=PARAM_GRID,
            cv=5,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        mlflow.log_metric("best_cv_f1_macro", grid.best_score_)
        for param, value in grid.best_params_.items():
            mlflow.log_param(f"best_{param}", value)

        test_score = grid.best_estimator_.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_score)

        # Log every individual candidate's result as a CSV artifact
        results_df = pd.DataFrame(grid.cv_results_)[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].sort_values("rank_test_score")
        results_df.to_csv("grid_search_all_candidates.csv", index=False)
        mlflow.log_artifact("grid_search_all_candidates.csv")

        print(f"Grid Search evaluated" 
              f"{total_combinations} combinations x 5 folds = "
              f"{total_combinations * 5} total fits"
        )

        print(
            f"Best params: {grid.best_params_}"
        )
        
        print(f"Best CV f1_macro: {grid.best_score_:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        return grid.best_score_, grid.best_params_


if __name__ == "__main__":
    run_grid_search("data/processed/iris_features.csv")

