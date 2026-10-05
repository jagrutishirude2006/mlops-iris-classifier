# src/register_best_model.py
"""
Queries MLflow for the best run in the experiment by f1_macro
and registers it under a named, versioned entry in the Model Registry.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient
# ---------------------------------------------------------
# 1. MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(
 os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
)
client = MlflowClient()
# ---------------------------------------------------------
# 2. Find the experiment
# ---------------------------------------------------------
experiment = client.get_experiment_by_name(
 "iris-classification-baseline"
)
if experiment is None:
 raise RuntimeError(
 "Experiment 'iris-classification-baseline' not found."
 )
experiment_id = experiment.experiment_id
print("Experiment ID:", experiment_id)
# ---------------------------------------------------------
# 3. Find the best run
# ---------------------------------------------------------
runs = client.search_runs(
 experiment_ids=[experiment_id],
 order_by=["metrics.f1_macro DESC"]
)
if not runs:
 raise RuntimeError("No runs found in the experiment.")
best_run = runs[0]
best_run_id = best_run.info.run_id
best_f1 = best_run.data.metrics["f1_macro"]
best_model_type = best_run.data.params["model_type"]
print("\nBest Run")
print("Run ID:", best_run_id)
print("Model:", best_model_type)
print("F1 Score:", best_f1)
# ---------------------------------------------------------
# 4. Register the best model
# ---------------------------------------------------------
model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "iris-classifier-prod"
print("\nRegistering model...")
print("Model URI:", model_uri)
model_version = mlflow.register_model(
 model_uri=model_uri,
 name=registered_model_name
)
print("\nModel Registered Successfully")
print("Model Name:", registered_model_name)
print("Version:", model_version.version)
# ---------------------------------------------------------
# 5. Transition model to Staging
# ---------------------------------------------------------
client.transition_model_version_stage(
 name=registered_model_name,
 version=model_version.version,
 stage="Staging"
)
print("\nModel moved to Staging successfully.")
print(
 f"Model URI: models:/{registered_model_name}/Staging"
)
