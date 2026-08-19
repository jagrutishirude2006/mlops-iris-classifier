"""
train.py
Baseline training script for the Iris classifier.
Used to demonstrate a version-controlled ML project structure.
"""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    iris = load_iris()
    return train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )


def train_model(X_train, y_train, n_estimators=100, max_depth=None):
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return acc, report


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    acc, report = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {acc:.4f}")
    print(report)

    joblib.dump(model, "models/iris_model.joblib")
    print("Model saved to models/iris_model.joblib")


if __name__ == "__main__":
    main()