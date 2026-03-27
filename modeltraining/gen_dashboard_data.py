import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split

# ── 1. CONFIG & REPRODUCIBILITY ─────────────────────────────────────────────
SEED = 42
DATA_PATH = "dataset/hyper_realistic_portfolio_100k.csv"
ARTIFACTS_DIR = "modeltraining"

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

def generate_dashboard_data():
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

    print("🚀 [Step 2] Selecting 30% Holdout Test Set...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    _, _, _, ids_test = train_test_split(X, cids, test_size=0.3, stratify=y, random_state=SEED)

    print("🚀 [Step 3] Predicting with Calibrated Ensemble...")
    p_xgb = m_xgb.predict_proba(X_test)[:, 1]
    p_lgbm = m_lgbm.predict_proba(X_test)[:, 1]
    X_meta = m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm}))
    p_final = m_meta.predict_proba(X_meta)[:, 1]
    
    bands = np.where(p_final >= t_red, 'RED', np.where(p_final >= t_green, 'YELLOW', 'GREEN'))

    # 4. Assemble Dashboard Source
    dashboard_df = X_test.copy()
    dashboard_df.insert(0, 'customer_id', ids_test.values)
    dashboard_df.insert(1, 'pd_final', p_final)
    dashboard_df.insert(2, 'risk_band', bands)
    dashboard_df['y_true'] = y_test.values
    
    dashboard_df.to_csv("dashboard_data.csv", index=False)
    print(f"✅ Created dashboard_data.csv with {len(dashboard_df)} records.")

if __name__ == "__main__":
    generate_dashboard_data()
