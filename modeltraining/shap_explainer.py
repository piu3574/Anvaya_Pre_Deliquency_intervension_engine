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

# ── 1. LOAD ARTIFACTS ───────────────────────────────────────────────────────
print("1. Loading Ensemble Artifacts...")
ARTIFACTS_DIR = "modeltraining"
lgbm_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))

with open(os.path.join(ARTIFACTS_DIR, 'woe_lookup.pkl'), 'rb') as f:
    woe_lookup = pickle.load(f)

with open(os.path.join(ARTIFACTS_DIR, 'banding_config.pkl'), 'rb') as f:
    banding_config = pickle.load(f)

# The meta-model PD is needed to filter customers
meta_model = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_meta.pkl"))

FEAT_COLS = list(woe_lookup.keys())

# Load some data to explain
df_raw = pd.read_csv("dataset/barclays_bank_synthetic_data.csv", nrows=10000)

# ── 2. FEATURE ENGINEERING & WoE ────────────────────────────────────────────
def apply_woe(df_raw):
    df = pd.DataFrame()
    df_days = pd.DataFrame()
    for i in range(1, 4):
        df_days[f'emi_payment_day_m{i}'] = pd.to_datetime(df_raw[f'emi_payment_day_m{i}']).dt.day
        
    df['F1_emi_to_income']     = (df_raw['total_monthly_emi_amount'] / (df_raw['monthly_net_salary'] + 1)).clip(0, 1.5)
    df['F2_savings_drawdown']  = (df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1)
    df['F3_salary_delay']      = (df_raw['expected_salary_day_of_month'] - pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day).fillna(0).abs()
    df['F4_spend_shift']       = (df_raw['total_debit_amount_30d'] / (df_raw['total_monthly_income'] + 1)).clip(0, 10)
    df['F5_auto_debit_fails']  = df_raw['failed_auto_debits_m1'] + df_raw['failed_auto_debits_m2']
    df['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d'].fillna(0)
    df['F7_overdraft_freq']    = df_raw['overdraft_days_30d']
    df['F8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    df['F9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    df['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    df['F10_peer_stress']      = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    df['F12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    df['F13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)

    df_woe = pd.DataFrame()
    for col in FEAT_COLS:
        lookup = woe_lookup[col]
        def map_val(v):
            for entry in lookup:
                if entry['bin'][0] <= v <= entry['bin'][1]: return entry['woe']
            return lookup[0]['woe'] if v < lookup[0]['bin'][0] else lookup[-1]['woe']
        df_woe[f"{col}_WoE"] = df[col].apply(map_val)
    return df_woe

print("2. Mapping Features to WoE...")
X_woe = apply_woe(df_raw)

# ── 3. SCORE & BAND ─────────────────────────────────────────────────────────
print("3. Scoring Customers for Filtering...")
pd_lgbm = lgbm_model.predict_proba(X_woe)[:, 1]
# We ideally use the real meta-score to decide whom to explain
X_meta = pd.DataFrame({'xgb': pd_lgbm, 'lgbm': pd_lgbm}) # Approximation for filtering
pd_final = meta_model.predict_proba(X_meta)[:, 1]

t_g = banding_config['green'] / 100
t_r = banding_config['red'] / 100
bands = np.where(pd_final < t_g, 'GREEN', np.where(pd_final < t_r, 'YELLOW', 'RED'))

# ── 4. SHAP (on LightGBM) ───────────────────────────────────────────────────
print("4. Building SHAP Explainer on LightGBM...")
explainer = shap.TreeExplainer(lgbm_model)
df_explain = X_woe[bands != 'GREEN'].copy()
indices = df_explain.index

if len(df_explain) > 0:
    print(f"5. Generating SHAP for {len(df_explain)} YELLOW/RED customers...")
    shap_results = explainer.shap_values(df_explain)
    
    # Handle SHAP return type (List vs Array)
    if isinstance(shap_results, list):
        shap_vals_class1 = shap_results[1]
    else:
        # If it's 3D [samples, features, outputs], take output 1
        if len(shap_results.shape) == 3:
            shap_vals_class1 = shap_results[:, :, 1]
        else:
            shap_vals_class1 = shap_results

    # ── 5. SAVE ─────────────────────────────────────────────────────────────────
    os.makedirs("explainability", exist_ok=True)
    out_path = "explainability/shap_explanations_ensemble.jsonl"
    print(f"6. Saving to {out_path}...")

    with open(out_path, 'w') as f:
        for i, idx in enumerate(indices):
            row_id = df_raw.iloc[idx]['customer_id']
            row_shap_v = shap_vals_class1[i]
            
            drivers = []
            for j, feat in enumerate(FEAT_COLS):
                val = float(row_shap_v[j])
                drivers.append({
                    "feature": feat,
                    "impact": val,
                    "direction": "RISK_UP" if val > 0 else "RISK_DOWN"
                })
            
            top_drivers = sorted(drivers, key=lambda x: abs(x['impact']), reverse=True)[:3]
            
            payload = {
                "customer_id": str(row_id),
                "pd_final": round(float(pd_final[idx]), 4),
                "risk_band": bands[idx],
                "top_drivers": top_drivers
            }
            f.write(json.dumps(payload) + "\n")
else:
    print("No non-GREEN customers found in sample.")

print("\n--- SHAP STAGE COMPLETE ---")
