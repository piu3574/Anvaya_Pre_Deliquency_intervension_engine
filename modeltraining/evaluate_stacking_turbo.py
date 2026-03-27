import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    accuracy_score, precision_recall_curve, auc
)
from scipy.stats import ks_2samp

# Config
DATA_PATH = "dataset/hyper_realistic_portfolio_100k.csv"
MODELS_DIR = "modeltraining"

def get_stats():
    # 1. Load Data
    df_raw = pd.read_csv(DATA_PATH)
    
    # Create the exact 10-feature signature expected by the 04:29 AM models
    X_turbo = pd.DataFrame({
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
    y = df_raw['default_status']
    
    _, X_test_raw, _, y_test = train_test_split(X_turbo, y, test_size=0.3, stratify=y, random_state=42)
    
    # 2. Load Models
    m_xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
    m_lgbm = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    m_meta = joblib.load(os.path.join(MODELS_DIR, "ensemble_meta.pkl"))
    m_scaler = joblib.load(os.path.join(MODELS_DIR, "meta_scaler.pkl"))
    woe = pickle.load(open(os.path.join(MODELS_DIR, "woe_lookup.pkl"), "rb"))
    
    # 3. Features Prep (Turbo uses direct stress vectors, skipping WoE)
    X_test_final = X_test_raw.copy()
    
    # Reorder to match XGB feature names
    xgb_feats = m_xgb.get_booster().feature_names
    X_test_final = X_test_final[xgb_feats]
    
    # 4. Predict
    pd_xgb = m_xgb.predict_proba(X_test_final)[:, 1]
    pd_lgbm = m_lgbm.predict_proba(X_test_final)[:, 1]
    
    X_meta = pd.DataFrame({'xgb': pd_xgb, 'lgbm': pd_lgbm})
    X_meta_s = m_scaler.transform(X_meta)
    pd_final = m_meta.predict_proba(X_meta_s)[:, 1]
    
    # 5. Metrics (Threshold 25% for RED)
    t_red = 0.25
    y_pred = (pd_final >= t_red).astype(int)
    
    auc_v = roc_auc_score(y_test, pd_final)
    gini = 2 * auc_v - 1
    ds = pd.DataFrame({'p': pd_final, 'y': y_test})
    ks = ks_2samp(ds[ds['y']==1]['p'], ds[ds['y']==0]['p']).statistic
    
    p, r, _ = precision_recall_curve(y_test, pd_final)
    pr_auc = auc(r, p)
    
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # Capture Rates
    total_bad = y_test.sum()
    ds_sort = ds.sort_values('p', ascending=False)
    c5 = ds_sort.head(int(len(ds_sort)*0.05))['y'].sum() / total_bad
    c10 = ds_sort.head(int(len(ds_sort)*0.10))['y'].sum() / total_bad
    
    print(f"AUC|{auc_v:.4f}")
    print(f"Gini|{gini:.4f}")
    print(f"KS|{ks:.4f}")
    print(f"PR-AUC|{pr_auc:.4f}")
    print(f"Precision|{prec:.4f}")
    print(f"Recall|{rec:.4f}")
    print(f"F1|{f1:.4f}")
    print(f"Accuracy|{acc:.4f}")
    print(f"Cap5|{c5:.4f}")
    print(f"Cap10|{c10:.4f}")

if __name__ == "__main__":
    get_stats()
