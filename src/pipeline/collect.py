# src/pipeline/collect.py
"""
Stage 1: Data Collection.
Simulates ingesting raw data from an external source and writing
it to the raw data zone, with basic collection-time metadata logging.
"""
import argparse
import logging
from datetime import datetime, timezone

import pandas as pd
from sklearn.datasets import load_iris

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("collect")


def collect_data(output_path: str) -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.rename(columns={"target": "species"})
    df["species"] = df["species"].map(dict(enumerate(iris.target_names)))
    df["collected_at"] = datetime.now(timezone.utc).isoformat()

    df.to_csv(output_path, index=False)
    logger.info("Collected %d rows -> %s", len(df), output_path)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", default="data/raw/iris_raw.csv")
    args = parser.parse_args()
    collect_data(args.output)
