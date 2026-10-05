# features.py
"""
Feature definitions for the Iris feature repository:
entity, data source, and feature view (schema + TTL).
"""
from datetime import timedelta
from feast import (Entity,FeatureView,Field,FileSource,FeatureService,)
from feast.types import Float32, String
from feast.value_type import ValueType

# ============================================================
# ENTITY
# ============================================================

sample = Entity(
    name="sample_id",
    join_keys=["sample_id"],
    value_type=ValueType.INT64,)

# ============================================================
# DATA SOURCE
# ============================================================

iris_source = FileSource(
    name="iris_features_source",
    path="data/iris_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)
# ============================================================
# FEATURE VIEW 1
# Original Iris measurements
# ============================================================
iris_measurements_fv = FeatureView(
    name="iris_measurements",
    entities=[sample],
    ttl=timedelta(days=365),

    schema=[
        Field(name="sepal length (cm)",dtype=Float32,),
        Field(name="sepal width (cm)",dtype=Float32,),
        Field(name="petal length (cm)",dtype=Float32,),
        Field(name="petal width (cm)",dtype=Float32,),
    ],

    source=iris_source,
    online=True,
)

# ============================================================
# FEATURE VIEW 2
# Engineered features
# ============================================================

iris_engineered_fv = FeatureView(
    name="iris_engineered_features",
    entities=[sample],
    ttl=timedelta(days=365),

    schema=[Field(name="sepal_area",dtype=Float32,),
        Field(
            name="petal_area",
            dtype=Float32,
        ),
        Field(
            name="sepal_to_petal_length_ratio",
            dtype=Float32,
        ),
        Field(
            name="petal_length_bin",
            dtype=String,
        ),
    ],

    source=iris_source,
    online=True,
)

# ============================================================
# FEATURE SERVICE
# ============================================================

iris_feature_service = FeatureService(
    name="iris_feature_service",
features=[
        iris_measurements_fv,
        iris_engineered_fv,
    ],
)