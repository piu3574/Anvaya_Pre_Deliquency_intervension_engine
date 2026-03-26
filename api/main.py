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

from api.supabase_client import get_raw_features  # Existing fetcher
from supabase import create_client, Client


# ── 0. Batch request model ──────────────────────────────────────────────────
class BatchRequest(BaseModel):
    customer_ids: List[str]


# ── 1. SUPABASE CLIENT (Service Role – currently hardcoded) ────────────────
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def save_score_to_supabase(
    ext_cust_id: str,
    p_xgb: float,
    p_lgbm: float,
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
            "pd_xgb": float(p_xgb),
            "pd_lgbm": float(p_lgbm),
            "pd_ensemble": float(p_ens),
            "risk_band": band,
            "t_green": float(t_g),
            "t_red": float(t_r),
            "source": "api_v1_ensemble",
        }
        res = supabase.table("risk_scores").insert(data).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as e:
        print(f"Supabase Log Error (Score): {e}")
        return None


def save_shap_to_supabase(risk_score_id: str, top_features: List[Dict]):
    if not supabase or not risk_score_id:
        return
    try:
        rows = []
        for i, feat in enumerate(top_features):
            rows.append(
                {
                    "risk_score_id": risk_score_id,
                    "feature_name": feat["feature"],
                    "shap_value": float(feat["value"]),
                    "rank": i + 1,
                }
            )
        supabase.table("risk_explanations").insert(rows).execute()
    except Exception as e:
        print(f"Supabase Log Error (SHAP): {e}")


app = FastAPI(title="Anvaya Pre-Delinquency Intervention Engine")


# ── 2. GLOBAL ARTIFACTS (Load on Startup) ──────────────────────────────────
ARTIFACTS_DIR = "modeltraining"
EXPLAIN_DIR = "explainability"

xgb_model = None
lgbm_model = None
woe_lookup = None
banding_config = None
shap_explainer = None

FEAT_COLS = [
    "F1_emi_to_income",
    "F2_savings_drawdown",
    "F3_salary_delay",
    "F4_spend_shift",
    "F5_auto_debit_fails",
    "F6_lending_app_usage",
    "F7_overdraft_freq",
    "F8_stress_velocity",
    "F9_payment_entropy",
    "F10_peer_stress",
    "F12_cross_loan",
    "F13_secondary_income",
    "F14_active_loan_pressure",
]


@app.on_event("startup")
async def load_artifacts():
    global xgb_model, lgbm_model
    global woe_lookup, banding_config, shap_explainer

    xgb_model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    lgbm_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "woe_lookup.pkl"), "rb") as f:
        woe_lookup = pickle.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "banding_config.pkl"), "rb") as f:
        banding_config = pickle.load(f)

    # Note: Using LGBM for SHAP as it's the dominant weight (0.6)
    shap_explainer = shap.TreeExplainer(lgbm_model)


# ── 3. SCORING UTILITIES ────────────────────────────────────────────────────
def get_woe(feat_name: str, value: float) -> float:
    lookup = woe_lookup.get(feat_name, [])
    for entry in lookup:
        if entry["bin"][0] <= value <= entry["bin"][1]:
            return entry["woe"]
    if lookup:
        return lookup[0]["woe"] if value < lookup[0]["bin"][0] else lookup[-1]["woe"]
    return 0.0


def compute_woe_features(raw_row: Dict[str, Any]) -> np.ndarray:
    """
    Orchestrates the transition from raw DB values to WoE vectors.
    1. Maps 'f1', 'f2' etc keys to the model's internal names.
    2. Ensures missing values default to 0.0.
    """
    woe_vec = []
    
    # Mapping dict: long name -> short DB key
    mapping = {
        "F1_emi_to_income": "f1",
        "F2_savings_drawdown": "f2",
        "F3_salary_delay": "f3",
        "F4_spend_shift": "f4",
        "F5_auto_debit_fails": "f5",
        "F6_lending_app_usage": "f6",
        "F7_overdraft_freq": "f7",
        "F8_stress_velocity": "f8",
        "F9_payment_entropy": "f9",
        "F10_peer_stress": "f10",
        "F12_cross_loan": "f12",
        "F13_secondary_income": "f13",
        "F14_active_loan_pressure": "f14"
    }
    
    for long_f in FEAT_COLS:
        short_key = mapping.get(long_f)
        # Attempt to get from DB via short key OR long key (fallback)
        val = raw_row.get(short_key) if short_key in raw_row else raw_row.get(long_f)
        
        # Default to 0.0 if missing (prevents static PD if keys mismatch)
        val = float(val) if val is not None else 0.0
        
        woe_vec.append(get_woe(long_f, val))
        
    return np.array(woe_vec)


def explain_customer(woe_features: np.ndarray) -> List[Dict]:
    sv_logit = shap_explainer.shap_values(woe_features.reshape(1, -1))[0]

    drivers = []
    for i, feat in enumerate(FEAT_COLS):
        val = float(sv_logit[i])
        drivers.append(
            {
                "feature": feat,
                "direction": "up" if val > 0 else "down",
                "value": abs(val),
                "reason_code": f"{feat.upper()}_{('HIGH' if val > 0 else 'LOW')}_RISK",
            }
        )
    return sorted(drivers, key=lambda x: x["value"], reverse=True)[:3]


# ── 4. ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}


@app.get("/score/{customer_id}")
async def get_score(customer_id: str):
    raw_row, error_msg = get_raw_features(customer_id)
    print(f"DEBUG: Customer {customer_id} raw_row (keys) = {list(raw_row.keys()) if raw_row else 'None'}")
    
    if error_msg:
        raise HTTPException(
            status_code=500, detail=f"Supabase Query Error: {error_msg}"
        )
    
    if not raw_row:
        raise HTTPException(
            status_code=404, detail=f"Customer {customer_id} not found in Supabase table 'customer_features'."
        )

    woe_vec = compute_woe_features(raw_row)
    print(f"DEBUG: WoE Vector for {customer_id} = {woe_vec}")
    
    # 4. Model Predictions & Simple Averaging
    pd_xgb = float(xgb_model.predict_proba(woe_vec.reshape(1, -1))[0, 1])
    pd_lgbm = float(lgbm_model.predict_proba(woe_vec.reshape(1, -1))[0, 1])

    # PD_final = 0.4 * XGB + 0.6 * LGBM (Per user request)
    pd_final = (0.4 * pd_xgb) + (0.6 * pd_lgbm)

    t_g = banding_config["green"] / 100
    t_r = banding_config["red"] / 100

    if pd_final < t_g:
        band = "GREEN"
    elif pd_final < t_r:
        band = "YELLOW"
    else:
        band = "RED"

    drivers = explain_customer(woe_vec)

    score_id = save_score_to_supabase(
        customer_id, pd_xgb, pd_lgbm, pd_final, band, t_g, t_r
    )
    if score_id:
        save_shap_to_supabase(score_id, drivers)

    return {
        "customer_id": customer_id,
        "score_id_logged": score_id,
        "pd_xgb": round(pd_xgb, 4),
        "pd_lgbm": round(pd_lgbm, 4),
        "pd_final": round(pd_final, 4),
        "band": band,
        "top_drivers": drivers,
        "model_version": "2.0.0_simple_averaging",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.post("/score/batch")
async def batch_score(request: BatchRequest):
    results = []
    for cid in request.customer_ids:
        try:
            res = await get_score(cid)
            results.append(res)
        except HTTPException:
            results.append(
                {
                    "customer_id": cid,
                    "status": "error",
                    "detail": "Not found",
                }
            )
    return results