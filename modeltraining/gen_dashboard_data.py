import pandas as pd
import numpy as np
import joblib
import pickle
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Paths
MODELS_DIR = "modeltraining"
DATA_PATH = "dataset/portfolio_100k.csv"

def generate_dashboard_data():
    print("1. Loading Data & Models...")
    df_raw = pd.read_csv(DATA_PATH)
    
    # Feature Engineering (Same as train_ensemble.py)
    df_feats = pd.DataFrame()
    df_feats['f1_emi_to_income']     = (df_raw['total_monthly_emi_amount'] / (df_raw['monthly_net_salary'] + 1)).clip(0, 1.5)
    df_feats['f2_savings_drawdown']  = (df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1)
    df_feats['f3_salary_delay']      = (df_raw['expected_salary_day_of_month'] - pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day).fillna(0).abs()
    df_feats['f4_spend_shift']       = (df_raw['total_debit_amount_30d'] / (df_raw['total_monthly_income'] + 1)).clip(0, 10)
    df_feats['f5_auto_debit_fails']  = df_raw['failed_auto_debits_m1'] + df_raw['failed_auto_debits_m2']
    df_feats['f6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d'].fillna(0)
    df_feats['f7_overdraft_freq']    = df_raw['overdraft_days_30d']
    df_feats['f8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    
    emi_cols = [f'emi_payment_day_m{i}' for i in range(1, 4)]
    df_days = pd.DataFrame()
    for c in emi_cols:
        df_days[c] = pd.to_datetime(df_raw[c]).dt.day
    df_feats['f9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    
    df_feats['f14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    df_feats['f10_peer_stress']      = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    df_feats['f12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    df_feats['f13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)
    
    y = df_raw['default_flag']
    customer_ids = df_raw['external_customer_id'] if 'external_customer_id' in df_raw.columns else [f"CUST_{i:06d}" for i in range(len(df_raw))]

    # 2. Split
    # We use random_state=42 consistent with train_ensemble.py to get the same 30% test set
    _, X_test, _, y_test = train_test_split(df_feats, y, test_size=0.3, stratify=y, random_state=42)
    _, ids_test, _, _    = train_test_split(customer_ids, y, test_size=0.3, stratify=y, random_state=42)

    # 3. WoE Transform
    with open(os.path.join(MODELS_DIR, "woe_lookup.pkl"), "rb") as f:
        woe_lookup = pickle.load(f)

    # Manual mapping from FE column to Lookup Key
    FE_TO_LOOKUP = {
        'f1_emi_to_income': 'F1_emi_to_income',
        'f2_savings_drawdown': 'F2_savings_drawdown',
        'f3_salary_delay': 'F3_salary_delay',
        'f4_spend_shift': 'F4_spend_shift',
        'f5_auto_debit_fails': 'F5_auto_debit_fails',
        'f6_lending_app_usage': 'F6_lending_app_usage',
        'f7_overdraft_freq': 'F7_overdraft_freq',
        'f8_stress_velocity': 'F8_stress_velocity',
        'f9_payment_entropy': 'F9_payment_entropy',
        'f10_peer_stress': 'F10_peer_stress',
        'f12_cross_loan': 'F12_cross_loan',
        'f13_secondary_income': 'F13_secondary_income',
        'f14_active_loan_pressure': 'F14_active_loan_pressure'
    }

    def get_woe(feat_name, val):
        lookup_key = FE_TO_LOOKUP.get(feat_name)
        if not lookup_key: return 0.0
        l = woe_lookup.get(lookup_key, [])
        for entry in l:
            if entry["bin"][0] <= val <= entry["bin"][1]:
                return entry["woe"]
        return l[0]["woe"] if l else 0.0

    X_test_woe = pd.DataFrame()
    for col in X_test.columns:
        X_test_woe[f"{FE_TO_LOOKUP[col]}_WoE"] = X_test[col].apply(lambda v: get_woe(col, v))

    # Ensure EXACT order and columns for Booster
    FEAT_ORDER = [
        'F1_emi_to_income_WoE', 'F2_savings_drawdown_WoE', 'F3_salary_delay_WoE',
        'F4_spend_shift_WoE', 'F5_auto_debit_fails_WoE', 'F6_lending_app_usage_WoE',
        'F7_overdraft_freq_WoE', 'F8_stress_velocity_WoE', 'F9_payment_entropy_WoE',
        'F10_peer_stress_WoE', 'F12_cross_loan_WoE', 'F13_secondary_income_WoE',
        'F14_active_loan_pressure_WoE'
    ]
    X_test_woe = X_test_woe[FEAT_ORDER]

    # 4. Predict
    m_xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    m_lgbm = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    
    pd_xgb = m_xgb.predict_proba(X_test_woe)[:, 1]
    pd_lgbm = m_lgbm.predict_proba(X_test_woe)[:, 1]
    pd_final = 0.4 * pd_xgb + 0.6 * pd_lgbm
    
    # 5. Banding (User Specified: 0.05 / 0.15)
    def get_band(p):
        if p < 0.15: return "GREEN"
        if p < 0.50: return "YELLOW"
        return "RED"
    
    bands = [get_band(p) for p in pd_final]

    # 6. Assemble Final Source of Truth
    dashboard_df = X_test.reset_index(drop=True)
    dashboard_df['customer_id'] = list(ids_test)
    dashboard_df['pd_xgb'] = pd_xgb
    dashboard_df['pd_lgbm'] = pd_lgbm
    dashboard_df['pd_final'] = pd_final
    dashboard_df['risk_band'] = bands
    dashboard_df['y_true'] = y_test.values

    # Clean headers (snake_case, no spaces - already done by my FE names f1_...)
    # Ensure customer_id is first
    cols = ['customer_id', 'pd_final', 'risk_band', 'y_true', 'pd_xgb', 'pd_lgbm'] + [c for c in X_test.columns]
    dashboard_df = dashboard_df[cols]

    dashboard_df.to_csv("dashboard_data.csv", index=False)
    print(f"--- SUCCESS: Generated dashboard_data.csv with {len(dashboard_df)} rows ---")

if __name__ == "__main__":
    generate_dashboard_data()
