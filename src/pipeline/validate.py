# src/pipeline/validate.py
"""
Stage 4: Data Validation.
Schema and statistical checks that must pass before data can
flow downstream to training. Raises on failure to hard-stop the pipeline.
"""
import argparse
import logging
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate")

EXPECTED_COLUMNS = {
    "sepal length (cm)", "sepal width (cm)",
    "petal length (cm)", "petal width (cm)",
    "species", "sepal_area", "petal_area",
    "sepal_to_petal_length_ratio", "petal_length_bin",
}
VALID_SPECIES = {"setosa", "versicolor", "virginica"}
RANGE_CHECKS = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.0),
}


class DataValidationError(Exception):
    pass


def validate(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    errors = []

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"Missing expected columns: {missing_cols}")

    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"Unexpected null values in columns: {null_cols}")

    invalid_species = set(df["species"].unique()) - VALID_SPECIES
    if invalid_species:
        errors.append(f"Unexpected species values: {invalid_species}")

    for col, (low, high) in RANGE_CHECKS.items():
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            errors.append(f"{len(out_of_range)} rows out of expected range for '{col}' ({low}-{high})")

    if errors:
        for e in errors:
            logger.error(e)
        raise DataValidationError(f"Validation failed with {len(errors)} error(s)")

    logger.info("Validation PASSED: %d rows, %d columns, all checks satisfied", len(df), df.shape[1])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/iris_features.csv")
    args = parser.parse_args()
    try:
        validate(args.input)
    except DataValidationError as e:
        logger.error("Pipeline halted: %s", e)
        sys.exit(1)
