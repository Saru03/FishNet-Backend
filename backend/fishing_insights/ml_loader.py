import os
import joblib
import xgboost as xgb
from huggingface_hub import hf_hub_download
from django.conf import settings
from earthaccess import login as earthaccess_login
import time
import threading
import gc

# Globals
_model = None
_scaler = None
_rf_imputer = None

_last_used = None
_idle_timeout = 600  # seconds (10 minutes)

# Thread safety lock for _last_used
_last_used_lock = threading.Lock()

CACHE_DIR = os.path.join(settings.BASE_DIR, ".hf-cache")
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_model.json')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml-models', 'pfz_scaler.pkl')

def get_last_used():
    with _last_used_lock:
        return _last_used

def update_last_used():
    with _last_used_lock:
        global _last_used
        _last_used = time.time()

def unload_if_idle():
    global _model, _scaler, _rf_imputer, _last_used
    while True:
        time.sleep(60)
        last_used_time = get_last_used()
        if last_used_time is not None:
            elapsed = time.time() - last_used_time
            print(f"DEBUG: Elapsed time since last use: {elapsed:.2f} seconds")
            if elapsed > _idle_timeout:
                print("DEBUG: Unloading ML models due to inactivity...")
                _model = None
                _scaler = None
                _rf_imputer = None
                gc.collect()
                with _last_used_lock:
                    _last_used = None
                print("DEBUG: ML models unloaded.")
        else:
            print("DEBUG: _last_used is None, models remain loaded.")

# Start the background thread
threading.Thread(target=unload_if_idle, daemon=True).start()

def get_ml_components():
    global _model, _scaler, _rf_imputer

    # Update last used timestamp
    update_last_used()

    if _model is None:
        print("DEBUG: Logging into Earthdata via environment...")
        auth = earthaccess_login(strategy="environment")
        print("Earthaccess login successful.")

        print("DEBUG: Loading ML components...")
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)

        _scaler = joblib.load(SCALER_PATH)

        rf_imputer_path = hf_hub_download(
            repo_id="saru03/Fishnet-Imputer",
            filename="rf_imputer_model.pkl",
            cache_dir=CACHE_DIR
        )
        _rf_imputer = joblib.load(rf_imputer_path)

        print("DEBUG: Model, scaler, imputer loaded.")

    return _model, _scaler, _rf_imputer