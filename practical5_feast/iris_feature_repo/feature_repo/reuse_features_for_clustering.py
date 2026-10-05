# reuse_features_for_clustering.py
"""
Demonstrates a completely different model (unsupervised clustering)
reusing the SAME registered engineered features without re-deriving them.
"""
from feast import FeatureStore
print("=" * 60)
print("FEAST FEATURE SERVICE TEST")
print("=" * 60)
store = FeatureStore(repo_path=".")
feature_service = store.get_feature_service(
    "iris_feature_service")
result = store.get_online_features(features=feature_service,
entity_rows=[{"sample_id": 1}],)
print("\nFeatures retrieved using Feature Service:")
print(result.to_dict())