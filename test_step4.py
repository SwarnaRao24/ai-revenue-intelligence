from pathlib import Path
import joblib

expected = [
    "models/churn_model.pkl",
    "models/churn_feature_cols.pkl",
    "models/revenue_model.pkl",
    "models/revenue_feature_cols.pkl",
    "models/segmentation_model.pkl",
    "models/segmentation_scaler.pkl",
    "models/anomaly_model.pkl",
    "models/anomaly_scaler.pkl",
]

print("\nChecking saved models...")
all_ok = True
for path in expected:
    exists = Path(path).exists()
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {path}")
    if not exists:
        all_ok = False

print("\nAll models saved!" if all_ok else "\nSome models are missing — rerun training.")