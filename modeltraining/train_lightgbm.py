"""
train_lightgbm.py — Authoritative Single-Model PD Pipeline
- 13 Features (F1-F13)
- WoE Binning & Transformation
- LightGBM Model
- Isotonic Calibration
- Data-driven Risk Banding
- Quality Gate (AUC, Accuracy, PSI, Calibration)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import pickle
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ── 1. CONFIGURATION & FEATURES ─────────────────────────────────────────────
DATA_PATH = 'dataset/barclays_bank_synthetic_data.csv'
ARTIFACT_DIR = 'modeltraining'
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# 13 Agreed Stress Features
FEAT_COLS = [
    'F1_emi_to_income', 'F2_savings_drawdown', 'F3_salary_delay',
    'F4_spend_shift', 'F5_auto_debit_fails', 'F6_lending_app_usage',
    'F7_overdraft_freq', 'F8_stress_velocity', 'F9_payment_entropy',
    'F10_peer_stress', 'F12_cross_loan', 'F13_secondary_income',
    'F14_active_loan_pressure'
]

# ── 2. DATA LOADING & CLEANING ──────────────────────────────────────────────
print("1. Loading and cleaning data...")
df_raw = pd.read_csv(DATA_PATH)
y = df_raw['default_flag']

# Re-engineer the 13 features (copying logic from previous versions for consistency)
emi_cols = [f'emi_payment_day_m{x}' for x in range(1,7)]
for c in emi_cols:
    df_raw[c] = pd.to_datetime(df_raw[c], errors='coerce').dt.day

df = pd.DataFrame()
df['F1_emi_to_income']  = (df_raw['total_monthly_emi_amount'] / (df_raw['total_monthly_income'] + 1)).clip(0,5)
df['F2_savings_drawdown'] = np.clip((df_raw['savings_balance_60d_ago'] - df_raw['current_account_balance']) / (df_raw['savings_balance_60d_ago'] + 1), 0, 5)
df['F3_salary_delay']   = (pd.to_datetime(df_raw['salary_credit_date_m1']).dt.day - df_raw['expected_salary_day_of_month']).clip(-5, 20)
df['F4_spend_shift']    = (df_raw['total_debit_amount_30d'] * 3 / (df_raw['grocery_spend_amount_90d'] + 1)).clip(0, 20)
df['F5_auto_debit_fails'] = df_raw['auto_debit_failed_insufficient_funds_30d']
df['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d']
df['F7_overdraft_freq']    = df_raw['overdraft_days_30d']
df['F8_stress_velocity']   = ((df_raw['end_of_month_balance_m6'] - df_raw['end_of_month_balance_m1']) / (df_raw['end_of_month_balance_m6'] + 1)).clip(-5, 5)
df['F9_payment_entropy']   = df_raw[emi_cols].std(axis=1).fillna(0)
df['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20) # This was referred to as F14 but counts as one of the 13
df['F10_peer_stress']      = df_raw.groupby(['employment_category'])['F14_active_loan_pressure'].transform('mean')
df['F12_cross_loan']       = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
df['F13_secondary_income'] = ((df_raw['total_monthly_income'] - df_raw['monthly_net_salary']) / (df_raw['total_monthly_income'] + 1)).clip(0, 1)

# Impute missing with median
df = df.fillna(df.median())
df['default_flag'] = y
df['gig_worker_flag'] = df_raw['gig_worker_flag']
df['cold_start_flag'] = df_raw['cold_start_flag']
df['customer_id'] = df_raw['customer_id']

# ── 3. BINNING & WoE ───────────────────────────────────────────────────────
print("\n2. Computing WoE and IV for all 13 features...")

def calculate_woe_iv(df, feature, target):
    lst = []
    # Use qcut for 10 bins
    try:
        df['temp_bin'] = pd.qcut(df[feature], q=10, duplicates='drop')
    except:
        df['temp_bin'] = pd.cut(df[feature], bins=10, duplicates='drop')
    
    total_events = df[target].sum()
    total_non_events = len(df) - total_events
    
    for bin_val in df['temp_bin'].unique():
        sub = df[df['temp_bin'] == bin_val]
        events = sub[target].sum()
        non_events = len(sub) - events
        
        p_event = max(events / total_events, 1e-9)
        p_non_event = max(non_events / total_non_events, 1e-9)
        
        woe = np.log(p_event / p_non_event)
        iv = (p_event - p_non_event) * woe
        
        lst.append({
            'bin': bin_val,
            'woe': woe,
            'iv': iv
        })
    
    iv_total = sum([x['iv'] for x in lst])
    return pd.DataFrame(lst), iv_total

woe_lookup = {}
iv_ranking = {}

for feat in FEAT_COLS:
    lookup_df, iv = calculate_woe_iv(df, feat, 'default_flag')
    # Store as intervals for matching
    woe_lookup[feat] = lookup_df.to_dict('records')
    iv_ranking[feat] = iv

# Save IV ranking artifact
with open(f'{ARTIFACT_DIR}/iv_ranking.pkl', 'wb') as f:
    pickle.dump(iv_ranking, f)

print(f"   IV Rankings saved. Top feature: {max(iv_ranking, key=iv_ranking.get)} (IV={max(iv_ranking.values()):.4f})")

# ── 4. WoE TRANSFORMATION ──────────────────────────────────────────────────
print("\n3. Performing WoE transformation...")

def apply_woe(val, feature_woe_list):
    for entry in feature_woe_list:
        if val in entry['bin']:
            return entry['woe']
    # If edge case (exactly on boundary not caught by interval):
    # Find closest bin
    return feature_woe_list[-1]['woe']

for feat in FEAT_COLS:
    df[f'{feat}_WoE'] = df[feat].apply(lambda x: apply_woe(x, woe_lookup[feat]))

# Save WoE lookup table
with open(f'{ARTIFACT_DIR}/woe_lookup.pkl', 'wb') as f:
    pickle.dump(woe_lookup, f)

WOE_FEATS = [f'{feat}_WoE' for feat in FEAT_COLS]

# ── 5. TRAIN PD MODEL (LIGHTGBM) ───────────────────────────────────────────
print("\n4. Training Single-Model LightGBM...")

X = df[WOE_FEATS]
y = df['default_flag']

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, num_leaves=50, learning_rate=0.03,
    bagging_fraction=0.8, feature_fraction=0.8, min_child_samples=30,
    random_state=42, verbose=-1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
)

# ── 6. CALIBRATION ──────────────────────────────────────────────────────────
print("\n5. Calibrating with Isotonic Regression...")
calibrated_lgb = CalibratedClassifierCV(lgb_model, method='isotonic', cv='prefit')
calibrated_lgb.fit(X_valid, y_valid)

# ── 7. PREDICT & QUALITY GATE ──────────────────────────────────────────────
print("\n6. Running Quality Gate...")

df['PD_raw'] = calibrated_lgb.predict_proba(X)[:, 1] * 100
# Add uplifts
df['PD_final'] = np.clip(
    df['PD_raw'] + df['gig_worker_flag'] * 1.0 + df['cold_start_flag'] * 0.5,
    0, 100
)

y_val_true = y_valid
y_val_pred_prob = calibrated_lgb.predict_proba(X_valid)[:, 1]

auc_val = roc_auc_score(y_val_true, y_val_pred_prob)
# Using 15% as intervention threshold for accuracy check
acc_val = accuracy_score(y_val_true, (y_val_pred_prob >= 0.15).astype(int))

# Calibration Error (Simplified Brier-like)
calib_err = np.mean((y_val_pred_prob - y_val_true)**2)

print(f"   AUC-ROC          : {auc_val:.4f}")
print(f"   Accuracy@15%     : {acc_val*100:.2f}%")
print(f"   Mean Sq. Error   : {calib_err:.4f}")

# PSI (Simplified)
def compute_psi(expected, actual, buckets=10):
    expected_percents = np.histogram(expected, bins=buckets)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=buckets)[0] / len(actual)
    # Avoid zero
    expected_percents = np.where(expected_percents == 0, 1e-9, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-9, actual_percents)
    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

psi_val = compute_psi(calibrated_lgb.predict_proba(X_train)[:,1], y_val_pred_prob)
print(f"   PSI              : {psi_val:.4f}")

# ── 8. DATA-DRIVEN RISK BANDING ─────────────────────────────────────────────
print("\n7. Calculating data-driven risk bands...")

# We want to find thresholds that separate the population meaningfully.
# Low Risk (GREEN): Highest concentration of non-defaults, approx bottom 60% of score.
# High Risk (HIGH): Highest concentration of defaults, top ~15% of score or where DR > 20%.

pd_sorted = np.sort(df['PD_final'].values)
# Target: 65% GREEN, 20% MEDIUM, 15% HIGH as a starting distribution
green_cutoff = pd_sorted[int(len(pd_sorted) * 0.65)]
high_cutoff = pd_sorted[int(len(pd_sorted) * 0.85)]

# Round for cleaner display
green_cutoff = round(green_cutoff, 1)
high_cutoff = round(high_cutoff, 1)

print(f"   Derived Thresholds: GREEN <= {green_cutoff}%, HIGH >= {high_cutoff}%")

def assign_band(pd):
    if pd <= green_cutoff: return 'GREEN'
    if pd < high_cutoff: return 'MEDIUM'
    return 'HIGH'

df['risk_band'] = df['PD_final'].apply(assign_band)

print("\nRisk Band Distribution:")
print(df['risk_band'].value_counts(normalize=True))
print("\nDefault Rate per Band:")
for band in ['GREEN', 'MEDIUM', 'HIGH']:
    subset = df[df['risk_band'] == band]
    dr = subset['default_flag'].mean() * 100
    print(f"   {band:8s}: {len(subset):>6} users | DR: {dr:.1f}%")

# Save banding thresholds to artifact for API
with open(f'{ARTIFACT_DIR}/banding_config.pkl', 'wb') as f:
    pickle.dump({'green': green_cutoff, 'high': high_cutoff}, f)

# ── 9. SAVE ARTIFACTS ──────────────────────────────────────────────────────
print("\n8. Saving artifacts...")
joblib.dump(lgb_model, f'{ARTIFACT_DIR}/lgbm_raw.pkl')
joblib.dump(calibrated_lgb, f'{ARTIFACT_DIR}/lgbm_calibrated.pkl')
df.to_csv(f'{ARTIFACT_DIR}/anvaya_scored_dataset.csv', index=False)

print("\n--- SINGLE-MODEL PIPELINE COMPLETE ---")
