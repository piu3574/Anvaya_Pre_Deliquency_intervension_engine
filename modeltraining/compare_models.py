import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# Config
DATA_PATH = "dataset/barclays_bank_synthetic_data.csv"
ARTIFACTS_DIR = "modeltraining"

print("1. Loading Data & Models...")
df_raw = pd.read_csv(DATA_PATH)
xgb = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
lgbm = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))
meta = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_meta.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "meta_scaler.pkl"))
banding = joblib.load(os.path.join(ARTIFACTS_DIR, "banding_config.pkl"))

with open(os.path.join(ARTIFACTS_DIR, "woe_lookup.pkl"), "rb") as f:
    woe_lookup = pickle.load(f)

# Splits
y = df_raw['default_flag']
_, temp_df, _, temp_y = train_test_split(df_raw, y, test_size=0.4, stratify=y, random_state=42)
_, test_df, _, y_test = train_test_split(temp_df, temp_y, test_size=0.5, stratify=temp_y, random_state=42)

# Engineering (Minimal for WoE)
def apply_woe(df_in):
    # F1-F14 mapping logic
    df = pd.DataFrame()
    df['F1_emi_to_income']     = (df_in['total_monthly_emi_amount'] / (df_in['monthly_net_salary'] + 1)).clip(0, 1.5)
    df['F2_savings_drawdown']  = (df_in['savings_balance_60d_ago'] - df_in['current_account_balance']) / (df_in['savings_balance_60d_ago'] + 1)
    df['F3_salary_delay']      = (df_in['expected_salary_day_of_month'] - pd.to_datetime(df_in['salary_credit_date_m1']).dt.day).fillna(0).abs()
    df['F4_spend_shift']       = (df_in['total_debit_amount_30d'] / (df_in['total_monthly_income'] + 1)).clip(0, 10)
    df['F5_auto_debit_fails']  = df_in['failed_auto_debits_m1'] + df_in['failed_auto_debits_m2']
    df['F6_lending_app_usage'] = df_in['lending_app_transaction_count_30d'].fillna(0)
    df['F7_overdraft_freq']    = df_in['overdraft_days_30d']
    df['F8_stress_velocity']   = ((df_in['end_of_month_balance_m6'] - df_in['end_of_month_balance_m1']) / (df_in['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    df_days = pd.DataFrame()
    for c in [f'emi_payment_day_m{i}' for i in range(1, 4)]:
        df_days[c] = pd.to_datetime(df_in[c]).dt.day
    df['F9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    df['F14_active_loan_pressure'] = (df_in['total_loan_outstanding'] / (df_in['total_credit_limit'] + 1)).clip(0, 20)
    df['F10_peer_stress']      = df_in.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_in['total_credit_limit'].mean() + 1)
    df['F12_cross_loan']       = df_in['number_of_active_loans'] / (df_in['customer_vintage_months'] + 1)
    df['F13_secondary_income'] = ((df_in['total_monthly_income'] - df_in['monthly_net_salary']) / (df_in['total_monthly_income'] + 1)).clip(0, 1)

    df_woe = pd.DataFrame()
    for col in woe_lookup.keys():
        lookup = woe_lookup[col]
        def map_val(v):
            for entry in lookup:
                if entry['bin'][0] <= v <= entry['bin'][1]: return entry['woe']
            return lookup[0]['woe'] if v < lookup[0]['bin'][0] else lookup[-1]['woe']
        df_woe[f"{col}_WoE"] = df[col].apply(map_val)
    return df_woe

print("2. Scoring Test Set...")
X_test_woe = apply_woe(test_df)
pd_xgb = xgb.predict_proba(X_test_woe)[:, 1]
pd_lgbm = lgbm.predict_proba(X_test_woe)[:, 1]

# Ensemble
X_meta = pd.DataFrame({'xgb': pd_xgb, 'lgbm': pd_lgbm})
X_meta_scaled = scaler.transform(X_meta)
pd_final = meta.predict_proba(X_meta_scaled)[:, 1]

t_r = banding['red'] / 100

print("\n--- PERFORMANCE SUMMARY (20k TEST SET) ---")
for name, p_scores in zip(["XGBoost", "LightGBM", "Ensemble"], [pd_xgb, pd_lgbm, pd_final]):
    auc = roc_auc_score(y_test, p_scores)
    # For individual accuracy, we'll use the same Red threshold as the ensemble comparison
    acc = accuracy_score(y_test, (p_scores >= t_r).astype(int))
    print(f"{name:10s} | AUC: {auc:.4f} | Accuracy: {acc:.4f}")
