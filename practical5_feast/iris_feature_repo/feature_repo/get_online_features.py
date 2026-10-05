# get_online_features.py
import pandas as pd
from feast import FeatureStore
print("=" * 60)
print("FEAST ONLINE FEATURE RETRIEVAL TEST")
print("=" * 60)
store = FeatureStore(repo_path=".")
result = store.get_online_features(
    features=[
"iris_measurements:sepal length (cm)",
        "iris_measurements:sepal width (cm)",
        "iris_measurements:petal length (cm)",
        "iris_measurements:petal width (cm)",
        "iris_engineered_features:sepal_area",
        "iris_engineered_features:petal_area",
        "iris_engineered_features:sepal_to_petal_length_ratio",
        "iris_engineered_features:petal_length_bin",
    ],
    entity_rows=[
        {"sample_id": 1}
    ],
)
result_dict = result.to_dict()
result_df = pd.DataFrame(result_dict)
print("\nFeatures retrieved for sample_id = 1:")
print(result_df.to_string(index=False))
