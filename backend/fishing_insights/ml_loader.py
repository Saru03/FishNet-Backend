# ml_loader.py
import os
import joblib
import xgboost as xgb
from earthaccess import login as earthaccess_login
from huggingface_hub import hf_hub_download
from django.conf import settings

CACHE_DIR = os.path.join(settings.BASE_DIR, ".hf-cache")
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_model.json')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_scaler.pkl')

# Initialize globals
_model = None
_scaler = None
_rf_imputer = None

def get_ml_components():
    global _model, _scaler, _rf_imputer

    if _model is None:
        # Optional: log into Earthdata (if needed)
        print("DEBUG: Logging into Earthdata via environment...")
        auth = earthaccess_login(strategy="environment")
        print("Earthaccess login successful.")

        # Load model
        print("DEBUG: Loading ML components...")
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)

        # Load scaler
        _scaler = joblib.load(SCALER_PATH)

        # Load imputer from HF Hub
        rf_imputer_path = hf_hub_download(
            repo_id="saru03/Fishnet-Imputer",
            filename="rf_imputer_model.pkl",
            cache_dir=CACHE_DIR
        )
        _rf_imputer = joblib.load(rf_imputer_path)

        print("DEBUG: Model, scaler, imputer loaded.")

    return _model, _scaler, _rf_imputer