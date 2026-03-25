import os
import joblib
import pickle
import json
import asyncio
import numpy as np
import pandas as pd
import shap
import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from api.supabase_client import get_raw_features, get_all_customers, log_score

app = FastAPI(title="Anvaya Pre-Delinquency Intervention Engine")

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 1. GLOBAL ARTIFACTS (Load on Startup) ──────────────────────────────────
ARTIFACTS_DIR = "modeltraining"

xgb_model = None
lgbm_model = None
meta_model = None
meta_scaler = None
woe_lookup = None
banding_config = None
shap_explainer = None

# Feature list (locked)
FEAT_COLS = [
    'F1_emi_to_income', 'F2_savings_drawdown', 'F3_salary_delay',
    'F4_spend_shift', 'F5_auto_debit_fails', 'F6_lending_app_usage',
    'F7_overdraft_freq', 'F8_stress_velocity', 'F9_payment_entropy',
    'F10_peer_stress', 'F12_cross_loan', 'F13_secondary_income',
    'F14_active_loan_pressure'
]

@app.on_event("startup")
async def load_artifacts():
    global xgb_model, lgbm_model, meta_model, meta_scaler, woe_lookup, banding_config, shap_explainer

    xgb_model  = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    lgbm_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))
    meta_model = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_meta.pkl"))
    meta_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "meta_scaler.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "woe_lookup.pkl"), "rb") as f:
        woe_lookup = pickle.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "banding_config.pkl"), "rb") as f:
        banding_config = pickle.load(f)

    shap_explainer = shap.TreeExplainer(lgbm_model)

# ── 2. SCORING UTILITIES ────────────────────────────────────────────────────
def get_woe(feat_name: str, value: float) -> float:
    lookup = woe_lookup.get(feat_name, [])
    for entry in lookup:
        if entry['bin'][0] <= value <= entry['bin'][1]:
            return entry['woe']
    if lookup:
        return lookup[0]['woe'] if value < lookup[0]['bin'][0] else lookup[-1]['woe']
    return 0.0

def compute_woe_features(raw_row: Dict[str, Any]) -> np.ndarray:
    woe_vec = []
    for f in FEAT_COLS:
        val = raw_row.get(f) or 0.0
        woe_vec.append(get_woe(f, val))
    return np.array(woe_vec)

def explain_customer(woe_features: np.ndarray) -> List[Dict]:
    sv_logit = shap_explainer.shap_values(woe_features.reshape(1, -1))[0]
    drivers = []
    for i, feat in enumerate(FEAT_COLS):
        val = float(sv_logit[i])
        drivers.append({
            "feature": feat,
            "direction": "up" if val > 0 else "down",
            "value": abs(val),
            "reason_code": f"{feat.upper()}_{'HIGH' if val > 0 else 'LOW'}_RISK"
        })
    return sorted(drivers, key=lambda x: x["value"], reverse=True)[:3]

def score_customer_internal(customer_id: str, raw_row: Dict[str, Any]) -> Dict:
    woe = compute_woe_features(raw_row)
    pd_xgb  = float(xgb_model.predict_proba(woe.reshape(1, -1))[0, 1])
    pd_lgbm = float(lgbm_model.predict_proba(woe.reshape(1, -1))[0, 1])

    meta_input = pd.DataFrame({'xgb': [pd_xgb], 'lgbm': [pd_lgbm]})
    meta_input_scaled = meta_scaler.transform(meta_input)
    pd_final = float(meta_model.predict_proba(meta_input_scaled)[0, 1])

    t_g = banding_config['green'] / 100
    t_r = banding_config['red'] / 100

    if pd_final < t_g:    band = "GREEN"
    elif pd_final < t_r:  band = "YELLOW"
    else:                  band = "RED"

    drivers = explain_customer(woe)
    reason_codes = [d["reason_code"] for d in drivers]

    return {
        "customer_id": customer_id,
        "pd_xgb": round(pd_xgb, 4),
        "pd_lgbm": round(pd_lgbm, 4),
        "pd_final": round(pd_final, 4),
        "band": band,
        "top_drivers": drivers,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# ── 4. ENDPOINTS ────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.get("/customers")
def list_customers():
    """Fetch all customers from Supabase customer_features table."""
    customers = get_all_customers()
    if customers is None:
        raise HTTPException(status_code=503, detail="Supabase not available.")
    return customers

@app.get("/score/{customer_id}")
async def get_score(customer_id: str):
    raw_row = get_raw_features(customer_id)
    if not raw_row:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found in Supabase.")

    result = score_customer_internal(customer_id, raw_row)
    log_score(customer_id, result["pd_final"], result["band"], [d["reason_code"] for d in result["top_drivers"]])
    return result

class BatchRequest(BaseModel):
    customer_ids: List[str]

@app.post("/score/batch")
async def batch_score(request: BatchRequest):
    results = []
    for cid in request.customer_ids:
        try:
            res = await get_score(cid)
            results.append(res)
        except HTTPException:
            results.append({"customer_id": cid, "status": "error", "detail": "Not found"})
    return results

@app.get("/pipeline/run")
async def run_pipeline():
    """
    SSE endpoint: scores all customers and streams log + score events.
    Each event is a JSON-encoded SSEMessage.
    """
    async def event_stream():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        yield sse({"type": "log", "message": "[INFO] Fetching customers from Supabase…"})
        await asyncio.sleep(0.05)

        customers = get_all_customers()
        if not customers:
            yield sse({"type": "error", "message": "Failed to fetch customers from Supabase."})
            return

        yield sse({"type": "log", "message": f"[INFO] Loaded {len(customers)} customers. Starting ensemble scoring…"})
        await asyncio.sleep(0.05)

        for i, customer in enumerate(customers):
            cid = customer.get("customer_id", f"cust_{i}")
            try:
                yield sse({"type": "log", "message": f"[{i+1}/{len(customers)}] Scoring {cid}…"})
                await asyncio.sleep(0.02)

                result = score_customer_internal(cid, customer)
                log_score(cid, result["pd_final"], result["band"], [d["reason_code"] for d in result["top_drivers"]])

                yield sse({"type": "score", "data": result})
                await asyncio.sleep(0.05)

            except Exception as e:
                yield sse({"type": "log", "message": f"[ERROR] {cid}: {str(e)}"})

        yield sse({"type": "done", "message": f"Pipeline complete. {len(customers)} customers scored."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
