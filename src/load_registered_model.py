# src/load_registered_model.py
import os
import mlflow
import mlflow.sklearn

# ---------------------------------------------------------
# 1. Configure MLflow tracking URI
# ---------------------------------------------------------


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
                        )

# ---------------------------------------------------------
# 2. Define registered model URI
# ---------------------------------------------------------

MODEL_URI = "models:/iris-classifier-prod/Staging"
print("=" * 60)
print("LOADING REGISTERED MODEL")
print("=" * 60)
print("Model URI:", MODEL_URI)

# ---------------------------------------------------------
# 3. Load model from Model Registry
# --------------------------------------------------------

model = mlflow.sklearn.load_model(MODEL_URI)
print("\nModel loaded successfully!")
print("Model type:", type(model))

# ---------------------------------------------------------
# 4. Display model
# --------------------------------------------------------

print("\nLoaded Model:")
print(model)
