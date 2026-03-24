from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Anvaya Single-Model PD API", version="1.0")

# ── 1. LOAD MODELS & ARTIFACTS ─────────────────────────────────────────────
MODELS_DIR = 'modeltraining'
try:
    lgbm_model = joblib.load(os.path.join(MODELS_DIR, 'lgbm_calibrated.pkl'))
    
    with open(os.path.join(MODELS_DIR, 'woe_lookup.pkl'), 'rb') as f:
        woe_lookup = pickle.load(f)
        
    with open(os.path.join(MODELS_DIR, 'banding_config.pkl'), 'rb') as f:
        banding_config = pickle.load(f)
        
    print(f"✅ Single-Model PD Pipeline loaded. GREEN <= {banding_config['green']}%, HIGH >= {banding_config['high']}%")
except Exception as e:
    print(f"⚠️ Error loading models/artifacts: {e}")

# ── 2. SCHEMA ──────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    F1_emi_to_income: float
    F2_savings_drawdown: float
    F3_salary_delay: float
    F4_spend_shift: float
    F5_auto_debit_fails: float
    F6_lending_app_usage: float
    F7_overdraft_freq: float
    F8_stress_velocity: float
    F9_payment_entropy: float
    F10_peer_stress: float
    F12_cross_loan: float
    F13_secondary_income: float
    F14_active_loan_pressure: float
    gig_worker_flag: int = 0
    cold_start_flag: int = 0

# ── 3. UTILITIES ────────────────────────────────────────────────────────────
def get_woe(feat_name, value):
    lookup = woe_lookup.get(feat_name, [])
    for entry in lookup:
        # entry['bin'] is a pandas Interval object (serialized as-is in pkl)
        if value in entry['bin']:
            return entry['woe']
    # Fallback to last bin if not matched (boundary conditions)
    if lookup:
        return lookup[-1]['woe']
    return 0.0

# ── 4. PREDICTION ENDPOINT ──────────────────────────────────────────────────
@app.post("/predict")
def predict_delinquency(data: CustomerData):
    feature_dict = data.dict()
    
    # 1. WoE Transformation
    woe_features = {}
    feats = [
        'F1_emi_to_income', 'F2_savings_drawdown', 'F3_salary_delay',
        'F4_spend_shift', 'F5_auto_debit_fails', 'F6_lending_app_usage',
        'F7_overdraft_freq', 'F8_stress_velocity', 'F9_payment_entropy',
        'F10_peer_stress', 'F12_cross_loan', 'F13_secondary_income',
        'F14_active_loan_pressure'
    ]
    
    for feat in feats:
        woe_features[f"{feat}_WoE"] = get_woe(feat, feature_dict[feat])
    
    # Ensure correct feature order for LightGBM
    x_input = pd.DataFrame([woe_features])[[f"{f}_WoE" for f in feats]]
    
    # 2. PD Prediction
    pd_raw = float(lgbm_model.predict_proba(x_input)[0, 1]) * 100
    
    # 3. Regulatory Uplifts
    pd_final = pd_raw + (data.gig_worker_flag * 1.0) + (data.cold_start_flag * 0.5)
    pd_final = min(100.0, max(0.0, pd_final))
    
    # 4. Data-Driven Risk Banding
    g_thresh = banding_config['green']
    h_thresh = banding_config['high']
    
    if pd_final <= g_thresh:
        band = 'GREEN'
    elif pd_final < h_thresh:
        band = 'MEDIUM'
    else:
        band = 'HIGH'
        
    return {
        "customer_id": "API_SCORE_REALTIME",
        "pd_raw": round(pd_raw, 2),
        "pd_final": round(pd_final, 2),
        "risk_band": band,
        "uplifts_applied": {
            "gig_worker": bool(data.gig_worker_flag),
            "cold_start": bool(data.cold_start_flag)
        },
        "thresholds": banding_config
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
