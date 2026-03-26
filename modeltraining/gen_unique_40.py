import pandas as pd
import numpy as np
import joblib
import json
import os
import pickle

# Paths
MODELS_DIR = "modeltraining"
DATA_PATH = "dataset/barclays_bank_synthetic_data.csv"

FEAT_ORDER = [
    'F1_emi_to_income_WoE', 'F2_savings_drawdown_WoE', 'F3_salary_delay_WoE',
    'F4_spend_shift_WoE', 'F5_auto_debit_fails_WoE', 'F6_lending_app_usage_WoE',
    'F7_overdraft_freq_WoE', 'F8_stress_velocity_WoE', 'F9_payment_entropy_WoE',
    'F10_peer_stress_WoE', 'F12_cross_loan_WoE', 'F13_secondary_income_WoE',
    'F14_active_loan_pressure_WoE'
]

def engineer_features(df_raw):
    df = pd.DataFrame()
    df['F1_emi_to_income']     = (df_raw['total_monthly_emi_amount'] / (df_raw['monthly_net_salary'] + 1)).clip(0, 1.5)
    df['F2_savings_drawdown']  = (df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1)
    df['F3_salary_delay']      = (df_raw['expected_salary_day_of_month'] - pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day).fillna(0).abs()
    df['F4_spend_shift']       = (df_raw['total_debit_amount_30d'] / (df_raw['total_monthly_income'] + 1)).clip(0, 10)
    df['F5_auto_debit_fails']  = df_raw['failed_auto_debits_m1'] + df_raw['failed_auto_debits_m2']
    df['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d'].fillna(0)
    df['F7_overdraft_freq']    = df_raw['overdraft_days_30d']
    df['F8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    
    emi_cols = [f'emi_payment_day_m{i}' for i in range(1, 4)]
    df_days = pd.DataFrame()
    for c in emi_cols:
        df_days[c] = pd.to_datetime(df_raw[c]).dt.day
    df['F9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    
    df['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    df['F10_peer_stress']      = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    df['F12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    df['F13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)
    
    return df

def run_unique_40():
    df_raw = pd.read_csv(DATA_PATH)
    sample = df_raw.sample(n=40, random_state=42)
    X_sample = engineer_features(sample)
    
    m_xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    m_lgbm = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    with open(os.path.join(MODELS_DIR, "woe_lookup.pkl"), "rb") as f:
        woe_lookup = pickle.load(f)

    def get_woe(feat_name, val, lookup_dict):
        l = lookup_dict.get(feat_name, [])
        for entry in l:
            if entry["bin"][0] <= val <= entry["bin"][1]:
                return entry["woe"]
        return l[0]["woe"] if l else 0.0

    X_woe = pd.DataFrame()
    # Explicitly map the raw col names to the WoE col names in FEAT_ORDER
    for col in X_sample.columns:
        X_woe[f"{col}_WoE"] = X_sample[col].apply(lambda v: get_woe(col, v, woe_lookup))
    
    # Reorder to match FEAT_ORDER
    X_woe = X_woe[FEAT_ORDER]
    
    pd_xgb = m_xgb.predict_proba(X_woe)[:, 1]
    pd_lgbm = m_lgbm.predict_proba(X_woe)[:, 1]
    pd_final = 0.4 * pd_xgb + 0.6 * pd_lgbm
    
    results = []
    for i, row_idx in enumerate(sample.index):
        p = round(float(pd_final[i]), 4)
        band = "GREEN" if p < 0.10 else ("YELLOW" if p < 0.20 else "RED")
        results.append({
            "id": f"TC{i+1:02d}",
            "xgb_pd": round(float(pd_xgb[i]), 4),
            "lgbm_pd": round(float(pd_lgbm[i]), 4),
            "pd_final": p,
            "band": band
        })
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_unique_40()
