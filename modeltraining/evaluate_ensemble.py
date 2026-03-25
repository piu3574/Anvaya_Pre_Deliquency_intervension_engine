"""
evaluate_ensemble.py — Full Performance Audit for Tier 1 & Tier 2 Metrics
- Handles AUC, KS, Gini, CM, Precision, Recall, Accuracy.
- Feature Importance (LightGBM).
- Lift, Calibration, and PSI.
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, brier_score_loss)
import warnings

warnings.filterwarnings('ignore')

# ── 1. CONFIG ───────────────────────────────────────────────────────────────
DATA_PATH = "dataset/barclays_bank_synthetic_data.csv"
ARTIFACTS_DIR = "modeltraining"
os.makedirs("docs", exist_ok=True)

# ── 2. UTILITIES & METRICS ──────────────────────────────────────────────────
def calculate_ks(y_true, y_prob):
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df = df.sort_values(by='p', ascending=False)
    df['good'] = 1 - df['y']
    df['bad'] = df['y']
    df['cum_good'] = df['good'].cumsum() / df['good'].sum()
    df['cum_bad'] = df['bad'].cumsum() / df['bad'].sum()
    ks = max(abs(df['cum_bad'] - df['cum_good']))
    return ks

def calculate_psi(expected, actual, bins=10):
    def get_buckets(data, bins):
        return pd.cut(data, bins=bins, labels=False, duplicates='drop')
    
    # Bucketize
    # Use quantile bins from expected to ensure consistency
    _, bin_edges = pd.qcut(expected, bins, duplicates='drop', retbins=True)
    expected_percents = pd.cut(expected, bin_edges, include_lowest=True).value_counts().sort_index()
    expected_percents = expected_percents / expected_percents.sum()
    
    actual_percents = pd.cut(actual, bin_edges, include_lowest=True).value_counts().sort_index()
    actual_percents = actual_percents / actual_percents.sum()
    
    # Smoothing
    expected_percents = expected_percents.replace(0, 0.0001)
    actual_percents = actual_percents.replace(0, 0.0001)
    
    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_value

# ── 3. DATA & MODEL LOADING ─────────────────────────────────────────────────
print("1. Loading Data and Replicating Splits...")
df_raw = pd.read_csv(DATA_PATH)

# Replicate engineering logic from train_ensemble.py
def engineer_features(df_raw):
    df = pd.DataFrame()
    y = df_raw['default_flag']
    df['F1_emi_to_income']     = (df_raw['total_monthly_emi_amount'] / (df_raw['monthly_net_salary'] + 1)).clip(0, 1.5)
    df['F2_savings_drawdown']  = (df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1)
    df['F3_salary_delay']      = (df_raw['expected_salary_day_of_month'] - pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day).fillna(0).abs()
    df['F4_spend_shift']       = (df_raw['total_debit_amount_30d'] / (df_raw['total_monthly_income'] + 1)).clip(0, 10)
    df['F5_auto_debit_fails']  = df_raw['failed_auto_debits_m1'] + df_raw['failed_auto_debits_m2']
    df['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d'].fillna(0)
    df['F7_overdraft_freq']    = df_raw['overdraft_days_30d']
    df['F8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
    df_days = pd.DataFrame()
    for c in [f'emi_payment_day_m{i}' for i in range(1, 4)]:
        df_days[c] = pd.to_datetime(df_raw[c]).dt.day
    df['F9_payment_entropy']   = df_days.std(axis=1).fillna(0)
    df['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    df['F10_peer_stress']      = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    df['F12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    df['F13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)
    return df, y

X, y = engineer_features(df_raw)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_test_feat, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Load Artifacts
xgb_model  = joblib.load(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"))
lgbm_model = joblib.load(os.path.join(ARTIFACTS_DIR, "lgbm_model.pkl"))
meta_model = joblib.load(os.path.join(ARTIFACTS_DIR, "ensemble_meta.pkl"))
meta_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "meta_scaler.pkl"))
with open(os.path.join(ARTIFACTS_DIR, "woe_lookup.pkl"), "rb") as f:
    woe_lookup = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "banding_config.pkl"), "rb") as f:
    banding_config = pickle.load(f)

# ── 4. TRANSFORMATION ───────────────────────────────────────────────────────
def apply_woe(df_in):
    df_out = pd.DataFrame()
    for col in woe_lookup.keys():
        lookup = woe_lookup[col]
        def map_val(v):
            for entry in lookup:
                if entry['bin'][0] <= v <= entry['bin'][1]: return entry['woe']
            return lookup[0]['woe'] if v < lookup[0]['bin'][0] else lookup[-1]['woe']
        df_out[f"{col}_WoE"] = df_in[col].apply(map_val)
    return df_out

X_test_woe = apply_woe(X_test)
X_train_woe = apply_woe(X_train)

# ── 5. SCORING ──────────────────────────────────────────────────────────────
print("2. Scoring Test Split (Scaled)...")
pd_xgb  = xgb_model.predict_proba(X_test_woe)[:, 1]
pd_lgbm = lgbm_model.predict_proba(X_test_woe)[:, 1]
X_meta_test = pd.DataFrame({'xgb': pd_xgb, 'lgbm': pd_lgbm})
X_meta_test_scaled = meta_scaler.transform(X_meta_test)
pd_final = meta_model.predict_proba(X_meta_test_scaled)[:, 1]

# Red Threshold for CM/Acc/Prec/Recall
t_r = banding_config['red'] / 100
y_pred = (pd_final >= t_r).astype(int)

# ── 6. TIER 1 METRICS ───────────────────────────────────────────────────────
print("3. Calculating Tier 1 Metrics...")
auc = roc_auc_score(y_test, pd_final)
ks = calculate_ks(y_test, pd_final)
gini = 2 * auc - 1
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

# Feature Importance
importances = lgbm_model.feature_importances_
feat_importance = pd.DataFrame({'feature': woe_lookup.keys(), 'importance': importances}).sort_values(by='importance', ascending=False)

# ── 7. TIER 2 METRICS ───────────────────────────────────────────────────────
print("4. Calculating Tier 2 Metrics...")
# Lift per Decile
test_results = pd.DataFrame({'y': y_test, 'p': pd_final})
test_results['decile'] = pd.qcut(test_results['p'], 10, labels=False, duplicates='drop')
overall_br = test_results['y'].mean()
decile_br = test_results.groupby('decile')['y'].mean()
lift = decile_br / overall_br

# Calibration
# Group by deciles and take mean pred vs mean actual
cal_expected = test_results.groupby('decile')['p'].mean()
cal_actual = test_results.groupby('decile')['y'].mean()

# PSI
# compare PD_final on test vs PD_final on train (proxy for dev stability)
pd_xgb_train = xgb_model.predict_proba(X_train_woe)[:, 1]
pd_lgbm_train = lgbm_model.predict_proba(X_train_woe)[:, 1]
pd_final_train = meta_model.predict_proba(pd.DataFrame({'xgb': pd_xgb_train, 'lgbm': pd_lgbm_train}))[:, 1]
psi = calculate_psi(pd_final_train, pd_final)

# ── 8. REPORT GENERATION ────────────────────────────────────────────────────
report = f"""# Ensemble PD Model Performance Report

## Tier 1: Core Performance
| Metric | Value |
| :--- | :--- |
| **AUC-ROC** | {auc:.4f} |
| **KS Statistic** | {ks:.4f} |
| **Gini Coefficient** | {gini:.4f} |
| **Accuracy** | {acc:.4f} |
| **Precision** | {prec:.4f} |
| **Recall** | {rec:.4f} |

### Confusion Matrix (at RED threshold t={t_r:.4f})
| | Pred 0 | Pred 1 |
| :--- | :--- | :--- |
| **Actual 0** | {tn} (TN) | {fp} (FP) |
| **Actual 1** | {fn} (FN) | {tp} (TP) |

### Top 5 Feature Importance (LightGBM Split)
{feat_importance.head(5).to_markdown(index=False)}

## Tier 2: Advanced Diagnostics
| Metric | Value |
| :--- | :--- |
| **PSI (Dev vs Test)** | {psi:.4f} |
| **Brier Score** | {brier_score_loss(y_test, pd_final):.4f} |

### Lift Chart (by Decile, 9=Riskiest)
{lift.to_markdown()}

### Calibration Curve Data
| Decile | Avg Predicted PD | Observed Default Rate |
| :--- | :--- | :--- |
"""
for i in range(len(cal_expected)):
    report += f"| {i} | {cal_expected.iloc[i]:.4f} | {cal_actual.iloc[i]:.4f} |\n"

with open("docs/model_performance_report.md", "w") as f:
    f.write(report)

print("\n--- PERFORMANCE EVALUATION COMPLETE ---")
print(f"Report saved to: docs/model_performance_report.md")
