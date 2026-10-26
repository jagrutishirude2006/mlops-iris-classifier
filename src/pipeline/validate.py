
# src/pipeline/validate.py

"""
Stage 4: Data Validation.
Checks the schema, missing values, species, and numeric ranges.
Raises DataValidationError if validation fails.
"""

import argparse
import logging
import sys

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("validate")


EXPECTED_COLUMNS = {
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "species",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
    "petal_length_bin",
}

VALID_SPECIES = {"setosa", "versicolor", "virginica"}

RANGE_CHECKS = {
    "sepal length (cm)": (3.0, 9.0),
    "sepal width (cm)": (1.5, 5.5),
    "petal length (cm)": (0.5, 8.0),
    "petal width (cm)": (0.05, 3.0),
}


class DataValidationError(Exception):
    """Raised when the input data fails validation."""


def validate(input_path: str) -> pd.DataFrame:
    """Read and validate a CSV file."""

    try:
        df = pd.read_csv(input_path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise DataValidationError(
            f"Unable to read input CSV: {exc}"
        ) from exc

    errors = []

    # 1. Validate required columns.
    missing_cols = EXPECTED_COLUMNS - set(df.columns)

    if missing_cols:
        errors.append(
            f"Missing expected columns: {sorted(missing_cols)}"
        )

    # 2. Validate species only if the column exists.
    if "species" in df.columns:
        invalid_species = (
            set(df["species"].dropna().unique()) - VALID_SPECIES
        )

        if invalid_species:
            errors.append(
                f"Unexpected species values: {sorted(invalid_species)}"
            )

    # 3. Check missing values.
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(
            f"Unexpected null values in columns: {null_cols}"
        )

    # 4. Validate numeric ranges.
    for col, (low, high) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue

        numeric_values = pd.to_numeric(
            df[col], errors="coerce"
        )

        # Detect non-numeric values.
        invalid_numeric = (
            numeric_values.isna() & df[col].notna()
        )

        if invalid_numeric.any():
            errors.append(
                f"Non-numeric values found in column '{col}'"
            )

        # Detect values outside the allowed range.
        out_of_range = (
            (numeric_values < low)
            | (numeric_values > high)
        ) & numeric_values.notna()

        if out_of_range.any():
            errors.append(
                f"{int(out_of_range.sum())} rows out of expected "
                f"range for '{col}' ({low}-{high})"
            )

    # 5. Raise detailed errors for pytest.
    if errors:
        for error in errors:
            logger.error(error)

        raise DataValidationError(
            "Validation failed: " + "; ".join(errors)
        )

    logger.info(
        "Validation PASSED: %d rows, %d columns, "
        "all checks satisfied",
        len(df),
        df.shape[1],
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the processed Iris dataset"
    )

    parser.add_argument(
        "--input",
        default="data/processed/iris_features.csv",
        help="Path to the input CSV file",
    )

    args = parser.parse_args()

    try:
        validate(args.input)
    except DataValidationError as exc:
        logger.error("Pipeline halted: %s", exc)
        sys.exit(1)
