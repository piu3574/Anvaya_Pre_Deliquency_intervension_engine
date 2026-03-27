import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os

def generate_scored_test_csv():
    print("1. Loading Data, Models & WoE Lookup...")
    df = pd.read_csv("dataset/portfolio_100k.csv")
    xgb = joblib.load("modeltraining/xgb_model.pkl")
    lgbm = joblib.load("modeltraining/lgbm_model.pkl")
    
    with open("modeltraining/woe_lookup.pkl", "rb") as f:
        woe_lookup = joblib.load(f)
        
    feat_mapping = {
        'total_monthly_emi_amount': 'F1_emi_to_income',
        'monthly_net_salary': 'F2_savings_drawdown', # Note: Using correct F-mapping from engineering
        'failed_auto_debits_m1': 'F5_auto_debit_fails',
        # ... and so on
    }

    # 2. Replicate the 30% Test Split (Total Split Logic)
    # The original split was 60/20/20 in train_ensemble.py
    # Train=60k, Val=20k, Test=20k.
    # User asked for "30k Test Case", so I'll follow the exact split logic from train_ensemble.py (40% split then 50% split)
    y = df['default_flag']
    _, df_temp = train_test_split(df, test_size=0.40, stratify=y, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp['default_flag'], random_state=42)
    
    # Wait, 100k * 0.40 * 0.50 = 20k.
    # User asked for 30k. I will provide the 30k split as per gen_dashboard_data.py
    _, df_test = train_test_split(df, test_size=0.30, stratify=y, random_state=42)

    print(f"2. Transforming and Scoring {len(df_test)} test cases...")
    
    # We need to compute features as per engineer_features() in train_ensemble.py
    df_raw = df_test
    X_raw = pd.DataFrame(index=df_test.index)
    X_raw['F1_emi_to_income']     = (df_raw['total_monthly_emi_amount'] / (df_raw['monthly_net_salary'] + 1)).clip(0, 1.5)
    X_raw['F2_savings_drawdown']  = (df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1)
    X_raw['F3_salary_delay']      = (df_raw['expected_salary_day_of_month'] - pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day).fillna(0).abs()
    X_raw['F4_spend_shift']       = (df_raw['total_debit_amount_30d'] / (df_raw['total_monthly_income'] + 1)).clip(0, 10)
    X_raw['F5_auto_debit_fails']  = df_raw['failed_auto_debits_m1'] + df_raw['failed_auto_debits_m2']
    X_raw['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d'].fillna(0)
    X_raw['F7_overdraft_freq']    = df_raw['overdraft_days_30d']
    X_raw['F8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    
    emi_cols = [f'emi_payment_day_m{i}' for i in range(1, 4)]
    df_days = pd.DataFrame()
    for c in emi_cols:
        df_days[c] = pd.to_datetime(df_raw[c]).dt.day
    X_raw['F9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    
    X_raw['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    X_raw['F10_peer_stress']      = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    
    X_raw['F12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    X_raw['F13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)

    FEAT_COLS = [
        "F1_emi_to_income","F2_savings_drawdown","F3_salary_delay","F4_spend_shift",
        "F5_auto_debit_fails","F6_lending_app_usage","F7_overdraft_freq","F8_stress_velocity",
        "F9_payment_entropy","F10_peer_stress","F12_cross_loan","F13_secondary_income",
        "F14_active_loan_pressure"
    ]

    # Apply WoE
    X_woe = pd.DataFrame(index=df_test.index)
    for feat in FEAT_COLS:
        lookup = woe_lookup.get(feat, [])
        def map_val(v):
            for entry in lookup:
                if entry["bin"][0] <= v <= entry["bin"][1]:
                    return entry["woe"]
            if lookup:
                return lookup[0]["woe"] if v < lookup[0]["bin"][0] else lookup[-1]["woe"]
            return 0.0
        X_woe[f"{feat}_WoE"] = X_raw[feat].apply(map_val)

    # Scoring
    pd_xgb = xgb.predict_proba(X_woe)[:, 1]
    pd_lgbm = lgbm.predict_proba(X_woe)[:, 1]
    pd_final = 0.4 * pd_xgb + 0.6 * pd_lgbm
    
    # Bands
    try:
        with open("modeltraining/optimized_thresholds.json", "r") as f:
            config = json.load(f)
            t_g = config.get("green_limit", 0.15)
            t_r = config.get("yellow_limit", 0.50)
    except:
        t_g, t_r = 0.15, 0.50

    bands = ["GREEN" if p < t_g else "YELLOW" if p < t_r else "RED" for p in pd_final]
    
    # Final Output
    result = df_test.copy()
    result['pd_xgb'] = pd_xgb
    result['pd_lgbm'] = pd_lgbm
    result['pd_final'] = pd_final
    result['risk_band'] = bands
    result.rename(columns={'default_flag': 'y_true'}, inplace=True)
    
    result.to_csv("anvaya_scored_test_30k.csv", index=False)
    print(f"✅ Exported 30,000 scored records.")

if __name__ == "__main__":
    generate_scored_test_csv()
