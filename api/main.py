import os
import joblib
import pickle
import numpy as np
import pandas as pd
import shap
import datetime
from fastapi import FastAPI, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from api.supabase_client import get_raw_features
from supabase import create_client, Client

# ── 0. Batch request model ──────────────────────────────────────────────────
class BatchRequest(BaseModel):
    customer_ids: List[str]

# ── 1. SUPABASE CLIENT ──────────────────────────────────────────────────────
# Note: Ideally these should be in .env. Hardcoded here for continuity with existing setup.
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def save_score_to_supabase(
    ext_cust_id: str,
    p_ens: float,
    band: str,
    t_g: float,
    t_r: float,
) -> Optional[str]:
    if not supabase:
        return None
    try:
        data = {
            "external_customer_id": ext_cust_id,
            "pd_ensemble": float(p_ens),
            "risk_band": band,
            "t_green": float(t_g),
            "t_red": float(t_r),
            "source": "api_v3_turbo_final",
        }
        res = supabase.table("risk_scores").insert(data).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as e:
        print(f"Supabase Log Error: {e}")
        return None

app = FastAPI(title="Anvaya Pre-Delinquency Engine (V3 Turbo)")

# ── 2. GLOBAL ARTIFACTS (V3 Turbo Final) ────────────────────────────────────
ARTIFACTS_DIR = os.path.join("modeltraining", "artifacts")

m_xgb = None
m_lgbm = None
m_meta = None
m_scaler = None
banding_config = None
shap_explainer = None

# Correct feature order for the Booster
FEATURE_NAMES = [
    'stress_f1', 'stress_f2', 'stress_f3', 'stress_f5', 'stress_f6', 
    'stress_f14', 'vintage', 'age', 'income_vol', 'overdraft'
]

@app.on_event("startup")
async def load_artifacts():
    global m_xgb, m_lgbm, m_meta, m_scaler, banding_config, shap_explainer
    
    try:
        m_xgb = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
        m_lgbm = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))
        m_meta = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_meta.pkl"))
        m_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "meta_scaler.pkl"))
        
        with open(os.path.join(ARTIFACTS_DIR, "banding_config.pkl"), "rb") as f:
            banding_config = pickle.load(f)
            
        # Using LGBM for SHAP as a representative base model
        shap_explainer = shap.TreeExplainer(m_lgbm)
        print("✅ V3 Turbo Artifacts Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# ── 3. FEATURE ENGINEERING & SCORING ────────────────────────────────────────
def engineer_turbo_vector(raw_row: Dict[str, Any]) -> pd.DataFrame:
    """
    Transforms raw DB fields into the 10-feature Decisive Signal set.
    Includes normalization (clipping) as per the Final Audit.
    """
    # Defensive floats with 0.0 defaults
    def d(k): return float(raw_row.get(k, 0.0) or 0.0)

    # Note: Mapping DB keys to the formulas in train_v3_turbo_final.py
    # stress_f1: EMI / Salary
    f1 = (d('total_monthly_emi_amount') / (d('total_salary_credit_30d') + 1))
    
    # stress_f2: Savings Drawdown
    f2 = (1 - (d('savings_balance_current') / (d('savings_balance_60d_ago') + 1)))
    
    # stress_f3: Salary Delay
    f3 = (abs(d('salary_credit_date_m1') - d('expected_salary_date')) / 10)
    
    # stress_f5: Auto-Debit Failure
    f5 = (d('auto_debit_failure_count_30d') / 5)
    
    # stress_f6: Lending App Activity
    f6 = (d('lending_app_transaction_count_30d') / 10)
    
    # stress_f14: Overleverage
    f14 = (d('total_loan_outstanding') / (d('total_credit_limit') + 1))
    
    # Demographics
    vin = d('customer_vintage_months') / 240.0
    age = d('age') / 75.0
    ivol = d('income_volatility_ratio_3m')
    ovd = d('overdraft_days_30d') / 30.0

    vec = {
        'stress_f1': np.clip(f1, 0, 1),
        'stress_f2': np.clip(f2, 0, 1),
        'stress_f3': np.clip(f3, 0, 1),
        'stress_f5': np.clip(f5, 0, 1),
        'stress_f6': np.clip(f6, 0, 1),
        'stress_f14': np.clip(f14, 0, 1),
        'vintage': vin,
        'age': age,
        'income_vol': ivol,
        'overdraft': ovd
    }
    return pd.DataFrame([vec])[FEATURE_NAMES]

def explain_customer(X_df: pd.DataFrame) -> List[Dict]:
    sv = shap_explainer.shap_values(X_df)[0]
    drivers = []
    for i, feat in enumerate(FEATURE_NAMES):
        val = float(sv[i])
        drivers.append({
            "feature": feat,
            "direction": "up" if val > 0 else "down",
            "impact": abs(val)
        })
    return sorted(drivers, key=lambda x: x["impact"], reverse=True)[:3]

# ── 4. ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "3.0.0-turbo", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.get("/score/{customer_id}")
async def get_score(customer_id: str):
    # 1. Fetch
    raw_row, error_msg = get_raw_features(customer_id)
    if error_msg: raise HTTPException(status_code=500, detail=error_msg)
    if not raw_row: raise HTTPException(status_code=404, detail="Customer not found")

    # 2. Engineer
    X_df = engineer_turbo_vector(raw_row)

    # 3. Predict Ensemble
    if m_xgb is None or m_lgbm is None or m_meta is None or m_scaler is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    p_xgb = m_xgb.predict_proba(X_df)[:, 1][0]
    p_lgbm = m_lgbm.predict_proba(X_df)[:, 1][0]
    
    # Meta-Feature Stacking
    X_meta = m_scaler.transform(pd.DataFrame({'xgb': [p_xgb], 'lgbm': [p_lgbm]}))
    p_final = float(m_meta.predict_proba(X_meta)[:, 1][0])

    t_g = banding_config["green"] / 100
    t_r = banding_config["red"] / 100

    if p_final < t_g: band = "GREEN"
    elif p_final < t_r: band = "YELLOW"
    else: band = "RED"

    # 4. Explain
    drivers = explain_customer(X_df)

    # 5. Log & Return
    score_id = save_score_to_supabase(customer_id, p_final, band, t_g, t_r)

    return {
        "customer_id": customer_id,
        "pd_final": round(p_final, 4),
        "risk_band": band,
        "top_drivers": drivers,
        "model_version": "3.0.0-turbo-calibrated",
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.post("/score/batch")
async def batch_score(request: BatchRequest):
    results = []
    for cid in request.customer_ids:
        try:
            res = await get_score(cid)
            results.append(res)
        except HTTPException as e:
            # Extract detail from HTTPException if available
            detail_message = e.detail if hasattr(e, 'detail') else "An error occurred"
            results.append(
                {
                    "customer_id": cid,
                    "status": "error",
                    "detail": detail_message,
                }
            )
    return results