"""
train_ensemble.py — Authoritative Ensemble PD Pipeline (XGB + LGB + Logistic Meta)
- 60/20/20 Stratified Split
- 13 WoE Features
- XGBoost & LightGBM Base Models
- Logistic Regression Meta-Model (Ensemble)
- Data-driven Banding Thresholds
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')

# ── 1. CONFIG ───────────────────────────────────────────────────────────────
DATA_PATH = "dataset/barclays_bank_synthetic_data.csv"
OUTPUT_DIR = "modeltraining"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed 13 Features (as per spec)
FEAT_COLS = [
    'F1_emi_to_income', 'F2_savings_drawdown', 'F3_salary_delay',
    'F4_spend_shift', 'F5_auto_debit_fails', 'F6_lending_app_usage',
    'F7_overdraft_freq', 'F8_stress_velocity', 'F9_payment_entropy',
    'F10_peer_stress', 'F12_cross_loan', 'F13_secondary_income',
    'F14_active_loan_pressure'
]

# ── 2. FEATURE ENGINEERING ──────────────────────────────────────────────────
def engineer_features(df_raw):
    df = pd.DataFrame()
    y = df_raw['default_flag']
    
    # F-Indicators (Mirroring validated logic from Phase-2)
    df['F1_emi_to_income']     = (df_raw['total_monthly_emi_amount'] / (df_raw['monthly_net_salary'] + 1)).clip(0, 1.5)
    df['F2_savings_drawdown']  = (df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1)
    df['F3_salary_delay']      = (df_raw['expected_salary_day_of_month'] - pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day).fillna(0).abs()
    df['F4_spend_shift']       = (df_raw['total_debit_amount_30d'] / (df_raw['total_monthly_income'] + 1)).clip(0, 10)
    df['F5_auto_debit_fails']  = df_raw['failed_auto_debits_m1'] + df_raw['failed_auto_debits_m2']
    df['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d'].fillna(0)
    df['F7_overdraft_freq']    = df_raw['overdraft_days_30d']
    df['F8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    
    emi_cols = [f'emi_payment_day_m{i}' for i in range(1, 4)]
    # Extract day from date strings for entropy calculation
    df_days = pd.DataFrame()
    for c in emi_cols:
        df_days[c] = pd.to_datetime(df_raw[c]).dt.day
    df['F9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    
    df['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    df['F10_peer_stress']      = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    
    df['F12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    df['F13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)

    return df, y

# ── 3. WoE CALCULATOR ───────────────────────────────────────────────────────
def calculate_woe_iv(df, y, feature):
    tmp = pd.DataFrame({'val': df[feature], 'y': y})
    # Quantile binning (10)
    tmp['bin'] = pd.qcut(tmp['val'], 10, duplicates='drop')
    
    stats = tmp.groupby('bin')['y'].agg(['count', 'sum'])
    stats['good'] = stats['count'] - stats['sum']
    stats['bad'] = stats['sum']
    
    # Smoothing
    stats['good'] = stats['good'].replace(0, 0.5)
    stats['bad'] = stats['bad'].replace(0, 0.5)
    
    p_good = stats['good'] / stats['good'].sum()
    p_bad = stats['bad'] / stats['bad'].sum()
    
    stats['woe'] = np.log(p_bad / p_good) # Bad / Good for risk
    return stats['woe'].to_dict()

# ── 4. MAIN FLOW ────────────────────────────────────────────────────────────
def run_training():
    print("1. Loading and Cleaning Data...")
    df_raw = pd.read_csv(DATA_PATH)
    X, y = engineer_features(df_raw)
    
    # 60/20/20 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"   Splits created: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. WoE Fitting (Train only)
    print("2. Fitting WoE Bins...")
    woe_lookup = {}
    for col in FEAT_COLS:
        # Save both bin edges and WoE
        bins = pd.qcut(X_train[col], 10, duplicates='drop', retbins=True)
        bin_edges = bins[1]
        
        # Calculate WoE
        tmp = pd.DataFrame({'val': X_train[col], 'y': y_train})
        tmp['bin_idx'] = pd.cut(tmp['val'], bins=bin_edges, include_lowest=True, labels=False)
        
        stats = tmp.groupby('bin_idx')['y'].agg(['count', 'sum'])
        stats['good'] = stats['count'] - stats['sum']
        stats['bad'] = stats['sum']
        stats['good'] = stats['good'].replace(0, 0.5)
        stats['bad'] = stats['bad'].replace(0, 0.5)
        
        p_good = stats['good'] / stats['good'].sum()
        p_bad = stats['bad'] / stats['bad'].sum()
        woe_map = np.log(p_bad / p_good)
        
        # Store for lookup
        lookup_entry = []
        for idx, w in woe_map.items():
            # Create interval object for lookup
            left = bin_edges[int(idx)]
            right = bin_edges[int(idx)+1]
            lookup_entry.append({'bin': (left, right), 'woe': w})
        
        woe_lookup[col] = lookup_entry

    # 6. Transform Splits
    def apply_woe(df_in):
        df_out = pd.DataFrame()
        for col in FEAT_COLS:
            lookup = woe_lookup[col]
            def map_val(v):
                for entry in lookup:
                    if entry['bin'][0] <= v <= entry['bin'][1]:
                        return entry['woe']
                # Out of range fallback
                if v < lookup[0]['bin'][0]: return lookup[0]['woe']
                return lookup[-1]['woe']
            df_out[f"{col}_WoE"] = df_in[col].apply(map_val)
        return df_out

    print("3. Transforming Datasets to WoE...")
    X_train_woe = apply_woe(X_train)
    X_val_woe = apply_woe(X_val)
    X_test_woe = apply_woe(X_test)
    
    WOE_COLS_INTERNAL = X_train_woe.columns.tolist()

    # 7. Base Model Training (Train)
    print("4. Training Base Models...")
    # XGBoost
    m_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric='logloss')
    m_xgb.fit(X_train_woe, y_train)
    
    # LightGBM
    m_lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    m_lgbm.fit(X_train_woe, y_train)

    # 8. Logistic Ensemble Training (Validation)
    print("5. Training Logistic Meta-Model (Validation)...")
    from sklearn.preprocessing import StandardScaler
    
    pd_xgb_val = m_xgb.predict_proba(X_val_woe)[:, 1]
    pd_lgbm_val = m_lgbm.predict_proba(X_val_woe)[:, 1]
    
    X_meta_val = pd.DataFrame({'xgb': pd_xgb_val, 'lgbm': pd_lgbm_val})
    
    # NEW: Standardization
    meta_scaler = StandardScaler()
    X_meta_val_scaled = meta_scaler.fit_transform(X_meta_val)
    
    # Meta-model with stronger regularization as per spec
    meta_model = LogisticRegression(penalty='l2', C=0.1, random_state=42)
    meta_model.fit(X_meta_val_scaled, y_val)
    
    # 9. Determine Thresholds (Validation) - DATA DRIVEN
    print("6. Calculating Data-Driven Thresholds (using Scaled Meta)...")
    pd_final_val = meta_model.predict_proba(X_meta_val_scaled)[:, 1]
    
    # Logic: 
    # GREEN: Target top 60% of cases or where DR <= 3%
    # RED: Target bottom 15% or where PD is extreme
    t_g = np.percentile(pd_final_val, 60) # Top 60% safe-ish
    t_r = np.percentile(pd_final_val, 85) # Bottom 15% risky

    # Refine based on observed bad rates if needed, but percentiles are a safe data-driven start
    print(f"   Derived Thresholds: GREEN <= {t_g:.4f}, RED >= {t_r:.4f}")

    # 10. Evaluation (Test)
    print("7. Final Quality Gate (Test)...")
    pd_xgb_test = m_xgb.predict_proba(X_test_woe)[:, 1]
    pd_lgbm_test = m_lgbm.predict_proba(X_test_woe)[:, 1]
    X_meta_test = pd.DataFrame({'xgb': pd_xgb_test, 'lgbm': pd_lgbm_test})
    X_meta_test_scaled = meta_scaler.transform(X_meta_test)
    
    pd_final_test = meta_model.predict_proba(X_meta_test_scaled)[:, 1]
    
    auc = roc_auc_score(y_test, pd_final_test)
    acc = accuracy_score(y_test, (pd_final_test > t_r).astype(int)) # Acc at RED threshold
    
    print(f"\nFinal Test Metrics (Scaled Ensemble):")
    print(f"   AUC-ROC          : {auc:.4f}")
    print(f"   Accuracy @ RED   : {acc:.4f}")
    
    # Banded Default Rates
    bands = np.where(pd_final_test < t_g, 'GREEN', np.where(pd_final_test < t_r, 'YELLOW', 'RED'))
    test_df = pd.DataFrame({'pd': pd_final_test, 'y': y_test, 'band': bands})
    
    for b in ['GREEN', 'YELLOW', 'RED']:
        subset = test_df[test_df['band'] == b]
        dr = subset['y'].mean() * 100 if len(subset) > 0 else 0
        print(f"   Band {b:6s} | Vol: {len(subset):6d} | DR: {dr:5.2f}%")

    # 11. PERSIST ARTIFACTS
    print("\n8. Saving Artifacts...")
    joblib.dump(m_xgb, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
    joblib.dump(m_lgbm, os.path.join(OUTPUT_DIR, "lgbm_model.pkl"))
    joblib.dump(meta_model, os.path.join(OUTPUT_DIR, "ensemble_meta.pkl"))
    joblib.dump(meta_scaler, os.path.join(OUTPUT_DIR, "meta_scaler.pkl"))
    
    with open(os.path.join(OUTPUT_DIR, "woe_lookup.pkl"), "wb") as f:
        pickle.dump(woe_lookup, f)
        
    banding_config = {
        'green': t_g * 100,
        'red': t_r * 100
    }
    with open(os.path.join(OUTPUT_DIR, "banding_config.pkl"), "wb") as f:
        pickle.dump(banding_config, f)

    # Save a small sample for baseline (PSI tracking)
    baseline = X_train_woe.mean().to_dict()
    with open(os.path.join(OUTPUT_DIR, "psi_baseline.pkl"), "wb") as f:
        pickle.dump(baseline, f)

    print("Pipeline Complete.")

if __name__ == "__main__":
    run_training()
