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

# Credentials
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Artifacts for V3 Turbo
MODELS_DIR = "modeltraining"
m_xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
m_lgbm = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
m_meta = joblib.load(os.path.join(MODELS_DIR, "ensemble_meta.pkl"))
m_scaler = joblib.load(os.path.join(MODELS_DIR, "meta_scaler.pkl"))

with open(os.path.join(MODELS_DIR, "banding_config.pkl"), "rb") as f:
    banding_config = pickle.load(f)

FEATURE_NAMES = [
    'stress_f1', 'stress_f2', 'stress_f3', 'stress_f5', 'stress_f6', 
    'stress_f14', 'vintage', 'age', 'income_vol', 'overdraft'
]

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "anvaya-dashboard-api", "version": "3.0.0-turbo"})

@app.route("/stats", methods=["GET"])
def stats():
    try:
        res = supabase.table("dashboard_customers").select("pd_final, risk_band", count="exact").execute()
        count = res.count
        data = res.data
        if not data: return jsonify({"count": 0, "msg": "No data in table"})
            
        df = pd.DataFrame(data)
        return jsonify({
            "total_count_in_db": count,
            "sample_count": len(df),
            "pd_final_stats": {
                "min": float(df["pd_final"].min()),
                "mean": float(df["pd_final"].mean()),
                "max": float(df["pd_final"].max())
            },
            "distribution_sample": df["risk_band"].value_counts().to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/customers", methods=["GET"])
def get_customers():
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    band = request.args.get("risk_band")
    query = supabase.table("dashboard_customers").select("*").range(offset, offset + limit - 1)
    if band: query = query.eq("risk_band", band.upper())
    res = query.execute()
    return jsonify(res.data)

@app.route("/predict", methods=["POST"])
def predict():
    req = request.json
    try:
        def d(k): return float(req.get(k, 0.0) or 0.0)

        # Turbo Feature Engineering
        f1 = (d('total_monthly_emi_amount') / (d('total_salary_credit_30d') + 1))
        f2 = (1 - (d('savings_balance_current') / (d('savings_balance_60d_ago') + 1)))
        f3 = (abs(d('salary_credit_date_m1') - d('expected_salary_date')) / 10)
        f5 = (d('auto_debit_failure_count_30d') / 5)
        f6 = (d('lending_app_transaction_count_30d') / 10)
        f14 = (d('total_loan_outstanding') / (d('total_credit_limit') + 1))
        
        vec = {
            'stress_f1': np.clip(f1, 0, 1),
            'stress_f2': np.clip(f2, 0, 1),
            'stress_f3': np.clip(f3, 0, 1),
            'stress_f5': np.clip(f5, 0, 1),
            'stress_f6': np.clip(f6, 0, 1),
            'stress_f14': np.clip(f14, 0, 1),
            'vintage': d('customer_vintage_months') / 240.0,
            'age': d('age') / 75.0,
            'income_vol': d('income_volatility_ratio_3m'),
            'overdraft': d('overdraft_days_30d') / 30.0
        }
        X_df = pd.DataFrame([vec])[FEATURE_NAMES]

        # Stacking Prediction
        p_xgb = m_xgb.predict_proba(X_df)[:, 1][0]
        p_lgbm = m_lgbm.predict_proba(X_df)[:, 1][0]
        X_meta = m_scaler.transform(pd.DataFrame({'xgb': [p_xgb], 'lgbm': [p_lgbm]}))
        p_final = float(m_meta.predict_proba(X_meta)[:, 1][0])

        t_g = banding_config["green"] / 100
        t_r = banding_config["red"] / 100
        band = "GREEN" if p_final < t_g else ("YELLOW" if p_final < t_r else "RED")
        
        return jsonify({
            "pd_final": round(p_final, 4),
            "risk_band": band,
            "version": "3.0.0-turbo"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
