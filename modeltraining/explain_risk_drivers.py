"""
shap_explainer.py — Authoritative Ensemble-Base SHAP Stage
- Loads Ensemble LGBM and WoE Lookup.
- Generates SHAP explanations for YELLOW and RED risk customers.
- Explanations are derived from the LightGBM base model (M_lgbm13).
"""
import pandas as pd
import numpy as np
import joblib
import shap
import json
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# ── 1. CONFIG & REPRODUCIBILITY ─────────────────────────────────────────────
SEED = 42
DATA_PATH = "dataset/hyper_realistic_portfolio_100k.csv"
ARTIFACTS_DIR = os.path.join("modeltraining", "artifacts")

FEATURE_NAMES = [
    'stress_f1', 'stress_f2', 'stress_f3', 'stress_f5', 'stress_f6', 
    'stress_f14', 'vintage', 'age', 'income_vol', 'overdraft'
]

def engineer_turbo_features(df_raw):
    """
    Same Turbo implementation as train_v3_turbo_final.py.
    """
    X = pd.DataFrame({
        'stress_f1': (df_raw['total_monthly_emi_amount'] / (df_raw['total_salary_credit_30d'] + 1)).clip(0, 1),
        'stress_f2': (1 - (df_raw['savings_balance_current'] / (df_raw['savings_balance_60d_ago'] + 1))).clip(0, 1),
        'stress_f3': ((df_raw['salary_credit_date_m1'] - df_raw['expected_salary_date']).abs() / 10).clip(0, 1),
        'stress_f5': (df_raw['auto_debit_failure_count_30d'] / 5).clip(0, 1),
        'stress_f6': (df_raw['lending_app_transaction_count_30d'] / 10).clip(0, 1),
        'stress_f14': (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 1),
        'vintage': df_raw['customer_vintage_months'] / 240.0,
        'age': df_raw['age'] / 75.0,
        'income_vol': df_raw['income_volatility_ratio_3m'],
        'overdraft': df_raw['overdraft_days_30d'] / 30.0
    })
    y = df_raw.get('default_status', 0)
    customer_ids = df_raw.get('external_customer_id', [f"CUST_{i:06d}" for i in range(len(df_raw))])
    return X, y, customer_ids

def run_shap_stage():
    print("🚀 [Step 1] Loading Hyper-Realistic Data & Models...")
    df_raw = pd.read_csv(DATA_PATH)
    X, y, cids = engineer_turbo_features(df_raw)
    
    m_xgb = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
    m_lgbm = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))
    m_meta = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_meta.pkl"))
    m_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "meta_scaler.pkl"))
    
    with open(os.path.join(ARTIFACTS_DIR, "banding_config.pkl"), "rb") as f:
        bc = pickle.load(f)
    t_green, t_red = bc['green']/100.0, bc['red']/100.0

    print("🚀 [Step 2] Filtering for High-Risk (YELLOW/RED) Customers...")
    p_xgb = m_xgb.predict_proba(X)[:, 1]
    p_lgbm = m_lgbm.predict_proba(X)[:, 1]
    X_meta = m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm}))
    p_final = m_meta.predict_proba(X_meta)[:, 1]
    
    bands = np.where(p_final >= t_red, 'RED', np.where(p_final >= t_green, 'YELLOW', 'GREEN'))
    high_risk_idx = np.where(bands != 'GREEN')[0]
    
    if len(high_risk_idx) == 0:
        print("No non-GREEN customers found. Terminating SHAP stage.")
        return

    # Subsample for speed if needed (e.g., first 500)
    sample_idx = high_risk_idx[:500] 
    X_sample = X.iloc[sample_idx]
    
    print(f"🚀 [Step 3] Building SHAP Explainer (on LightGBM base)...")
    explainer = shap.TreeExplainer(m_lgbm)
    shap_results = explainer.shap_values(X_sample)
    
    # Handle SHAP return type
    if isinstance(shap_results, list): shap_vals = shap_results[1]
    elif len(shap_results.shape) == 3: shap_vals = shap_results[:, :, 1]
    else: shap_vals = shap_results

    # 4. Save
    os.makedirs("explainability", exist_ok=True)
    out_path = "explainability/shap_explanations_ensemble.jsonl"
    print(f"🚀 [Step 4] Saving to {out_path}...")

    with open(out_path, 'w') as f:
        for i, idx in enumerate(sample_idx):
            row_id = cids[idx]
            row_shap_v = shap_vals[i]
            
            drivers = []
            for j, feat in enumerate(FEATURE_NAMES):
                val = float(row_shap_v[j])
                drivers.append({
                    "feature": feat,
                    "impact": val,
                    "direction": "RISK_UP" if val > 0 else "RISK_DOWN"
                })
            
            top_drivers = sorted(drivers, key=lambda x: abs(x['impact']), reverse=True)[:3]
            
            payload = {
                "customer_id": str(row_id),
                "pd_final": round(float(p_final[idx]), 4),
                "risk_band": bands[idx],
                "top_drivers": top_drivers
            }
            f.write(json.dumps(payload) + "\n")
    print("No non-GREEN customers found in sample.")

print("\n--- SHAP STAGE COMPLETE ---")
