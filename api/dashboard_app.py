from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
import os
import joblib
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)

# Credentials (Hardcoded as per existing pattern for immediate implementation)
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Artifacts for /predict
MODELS_DIR = "modeltraining"
xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
lgbm_model = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
with open(os.path.join(MODELS_DIR, "woe_lookup.pkl"), "rb") as f:
    woe_lookup = pickle.load(f)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "anvaya-dashboard-api"})

@app.route("/stats", methods=["GET"])
def stats():
    try:
        # Get count and aggregate stats
        res = supabase.table("dashboard_customers").select("pd_final, risk_band", count="exact").execute()
        count = res.count
        data = res.data
        if not data:
            return jsonify({"count": 0, "msg": "No data in table"})
            
        df = pd.DataFrame(data)
        stats_val = {
            "total_count_in_db": count,
            "sample_count": len(df),
            "pd_final_stats": {
                "min": float(df["pd_final"].min()),
                "mean": float(df["pd_final"].mean()),
                "max": float(df["pd_final"].max())
            },
            "distribution_sample": df["risk_band"].value_counts().to_dict()
        }
        return jsonify(stats_val)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/customers", methods=["GET"])
def get_customers():
    # Pagination & Filtering
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    band = request.args.get("risk_band")
    
    query = supabase.table("dashboard_customers").select("*").range(offset, offset + limit - 1)
    if band:
        query = query.eq("risk_band", band.upper())
        
    res = query.execute()
    return jsonify(res.data)

@app.route("/predict", methods=["POST"])
def predict():
    req = request.json
    # Expecting raw features f1...f14
    # Just a simple implementation for foundation
    try:
        raw_vals = []
        FE_MAP = [
            ('f1_emi_to_income', 'F1_emi_to_income'),
            ('f2_savings_drawdown', 'F2_savings_drawdown'),
            ('f3_salary_delay', 'F3_salary_delay'),
            ('f4_spend_shift', 'F4_spend_shift'),
            ('f5_auto_debit_fails', 'F5_auto_debit_fails'),
            ('f6_lending_app_usage', 'F6_lending_app_usage'),
            ('f7_overdraft_freq', 'F7_overdraft_freq'),
            ('f8_stress_velocity', 'F8_stress_velocity'),
            ('f9_payment_entropy', 'F9_payment_entropy'),
            ('f10_peer_stress', 'F10_peer_stress'),
            ('f12_cross_loan', 'F12_cross_loan'),
            ('f13_secondary_income', 'F13_secondary_income'),
            ('f14_active_loan_pressure', 'F14_active_loan_pressure')
        ]
        
        # 1. Transform to WoE
        woe_features = []
        for raw_k, lookup_k in FE_MAP:
            val = float(req.get(raw_k, 0.0))
            # WoE Lookup
            lookup = woe_lookup.get(lookup_k, [])
            found = False
            for entry in lookup:
                if entry["bin"][0] <= val <= entry["bin"][1]:
                    woe_features.append(entry["woe"])
                    found = True
                    break
            if not found:
                woe_features.append(lookup[0]["woe"] if lookup else 0.0)
        
        woe_vec = np.array(woe_features).reshape(1, -1)
        
        # 2. Predict
        p_xgb = float(xgb_model.predict_proba(woe_vec)[0, 1])
        p_lgbm = float(lgbm_model.predict_proba(woe_vec)[0, 1])
        p_final = (0.4 * p_xgb) + (0.6 * p_lgbm)
        
        # 3. Banding (5/15 Thresholds)
        band = "GREEN" if p_final < 0.05 else ("YELLOW" if p_final < 0.15 else "RED")
        
        return jsonify({
            "pd_xgb": round(p_xgb, 4),
            "pd_lgbm": round(p_lgbm, 4),
            "pd_final": round(p_final, 4),
            "risk_band": band
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Use different port to avoid conflict with main FastAPI app (8000)
    app.run(host="0.0.0.0", port=8001, debug=True)
