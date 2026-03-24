"""
pipeline_tests.py
Full end-to-end test suite for Anvaya Phase-2 pipeline.
Tests: edge cases, banding correctness, monotonicity, performance spot-check.
"""
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS & ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
xgb_model = joblib.load('modeltraining/xgb_fastpath.pkl')
lgb_model  = joblib.load('modeltraining/lgbm_calibrated.pkl')
with open('modeltraining/iv_ranking.pkl', 'rb') as f:
    iv_data = pickle.load(f)
top4 = [f"{feat}_WoE" for feat in iv_data['top4']]

ALL_WOE = [
    'F1_emi_to_income_WoE','F2_savings_drawdown_WoE','F3_salary_delay_WoE',
    'F4_spend_shift_WoE','F5_auto_debit_fails_WoE','F6_lending_app_usage_WoE',
    'F7_overdraft_freq_WoE','F8_stress_velocity_WoE','F9_payment_entropy_WoE',
    'F10_peer_stress_WoE','F12_cross_loan_WoE','F13_secondary_income_WoE',
    'F14_active_loan_pressure_WoE'
]

# Load WoE lookup
with open('modeltraining/woe_lookup.pkl', 'rb') as f:
    woe_lookup_raw = pickle.load(f)

# Load a sample of training data to build bin boundaries for WoE mapping
df_train = pd.read_csv('dataset/barclays_bank_synthetic_data.csv', nrows=50000)
y_train = df_train['default_flag']

emi_cols = [f'emi_payment_day_m{x}' for x in range(1,7)]
for c in emi_cols:
    df_train[c] = pd.to_datetime(df_train[c], errors='coerce').dt.day

df_train['F1_emi_to_income'] = (df_train['total_monthly_emi_amount'] / (df_train['total_monthly_income'] + 1)).clip(0, 5)
df_train['F2_savings_drawdown'] = np.clip((df_train['savings_balance_60d_ago'] - df_train['current_account_balance']) / (df_train['savings_balance_60d_ago'] + 1), 0, 5)
try:
    df_train['F3_salary_delay'] = (pd.to_datetime(df_train['salary_credit_date_m1']).dt.day - df_train['expected_salary_day_of_month']).clip(-5, 20)
except:
    df_train['F3_salary_delay'] = 0
df_train['F4_spend_shift'] = (df_train['total_debit_amount_30d'] * 3 / (df_train['grocery_spend_amount_90d'] + 1)).clip(0, 20)
df_train['F5_auto_debit_fails'] = df_train['auto_debit_failed_insufficient_funds_30d']
df_train['F6_lending_app_usage'] = df_train['lending_app_transaction_count_30d']
df_train['F7_overdraft_freq'] = df_train['overdraft_days_30d']
df_train['F8_stress_velocity'] = ((df_train['end_of_month_balance_m6'] - df_train['end_of_month_balance_m1']) / (df_train['end_of_month_balance_m6'] + 1)).clip(-5, 5)
df_train['F9_payment_entropy'] = df_train[emi_cols].std(axis=1).fillna(0)
df_train['F14_active_loan_pressure'] = (df_train['total_loan_outstanding'] / (df_train['total_credit_limit'] + 1)).clip(0, 20)
df_train['F10_peer_stress'] = df_train.groupby(['employment_category'])['F14_active_loan_pressure'].transform('mean')
df_train['F12_cross_loan'] = df_train['number_of_active_loans'] / (df_train['customer_vintage_months'] + 1)
df_train['F13_secondary_income'] = ((df_train['total_monthly_income'] - df_train['monthly_net_salary']) / (df_train['total_monthly_income'] + 1)).clip(0, 1)

FEAT_COLS = ['F1_emi_to_income','F2_savings_drawdown','F3_salary_delay','F4_spend_shift',
             'F5_auto_debit_fails','F6_lending_app_usage','F7_overdraft_freq','F8_stress_velocity',
             'F9_payment_entropy','F10_peer_stress','F12_cross_loan','F13_secondary_income','F14_active_loan_pressure']

# Build bin structures from training data
bin_structures = {}
for feat in FEAT_COLS:
    vals = df_train[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
    try:    bins_obj = pd.qcut(vals, q=10, duplicates='drop')
    except: bins_obj = pd.cut(vals, bins=10, duplicates='drop')
    bin_structures[feat] = bins_obj.cat.categories

    total_d = y_train.sum(); total_s = (y_train == 0).sum()
    woe_d = {}
    for bl in bins_obj.cat.categories:
        mask = (bins_obj == bl)
        d_b = y_train[mask].sum(); s_b = (y_train == 0)[mask].sum()
        p_d = max(d_b/total_d, 1e-9); p_s = max(s_b/total_s, 1e-9)
        woe_d[str(bl)] = np.log(p_d/p_s)
    woe_lookup_raw[feat] = woe_d  # Refresh with training-built bins

def map_to_woe(feat, value):
    """Map a raw feature value to WoE using training-built bin boundaries."""
    cats = bin_structures[feat]
    for interval in cats:
        if interval.left < value <= interval.right:
            return woe_lookup_raw[feat].get(str(interval), 0.0)
    # Edge: below min bin
    if value <= cats[0].right:
        return woe_lookup_raw[feat].get(str(cats[0]), 0.0)
    # Edge: above max bin
    return woe_lookup_raw[feat].get(str(cats[-1]), 0.0)

def score_customer(feat_dict, gig_worker=0, cold_start=0, label=""):
    """Run a single customer through the full dual-track pipeline."""
    # WoE transform
    woe_row = {}
    for feat in FEAT_COLS:
        woe_row[f'{feat}_WoE'] = map_to_woe(feat, feat_dict.get(feat, 0.0))

    x_xgb = pd.DataFrame([woe_row])[top4]
    pd_xgb = float(xgb_model.predict_proba(x_xgb)[0, 1]) * 100

    # Triage
    if pd_xgb <= 5.0:   triage = 'GREEN'
    elif pd_xgb >= 20.0: triage = 'RED_FAST'
    else:               triage = 'BORDERLINE'

    pd_lgb    = None
    gig_up    = gig_worker * 1.0
    cold_up   = cold_start * 0.5

    if triage == 'BORDERLINE':
        x_lgb  = pd.DataFrame([woe_row])[ALL_WOE]
        pd_lgb = float(lgb_model.predict_proba(x_lgb)[0, 1]) * 100
        pd_final = 0.4 * pd_xgb + 0.6 * pd_lgb
    else:
        pd_final = pd_xgb

    pd_final = np.clip(pd_final + gig_up + cold_up, 0, 100)

    if triage == 'GREEN':           band = 'GREEN'
    elif pd_final < 15.0:           band = 'MEDIUM'
    else:                           band = 'HIGH'

    return {
        'Case': label,
        'PD_XGB (%)': round(pd_xgb, 2),
        'PD_LGB (%)': round(pd_lgb, 2) if pd_lgb else '—',
        'Triage':  triage,
        'Gig Up (pp)': gig_up,
        'Cold Up (pp)': cold_up,
        'PD_Final (%)': round(pd_final, 2),
        'Band': band
    }

# ═════════════════════════════════════════════════════════════
# TEST 1: EDGE CASE SYNTHETIC CUSTOMERS
# ═════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: SYNTHETIC EDGE CASE CUSTOMERS")
print("=" * 70)

LOW_RISK = dict(
    F1_emi_to_income=0.10, F2_savings_drawdown=0.0, F3_salary_delay=0,
    F4_spend_shift=1.0, F5_auto_debit_fails=0, F6_lending_app_usage=0,
    F7_overdraft_freq=0, F8_stress_velocity=0.3, F9_payment_entropy=0.5,
    F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0,
    F14_active_loan_pressure=0.2)

MEDIUM_RISK = dict(
    F1_emi_to_income=0.42, F2_savings_drawdown=0.25, F3_salary_delay=4,
    F4_spend_shift=3.5, F5_auto_debit_fails=1, F6_lending_app_usage=1,
    F7_overdraft_freq=8, F8_stress_velocity=-0.10, F9_payment_entropy=5.0,
    F10_peer_stress=0.9, F12_cross_loan=0.4, F13_secondary_income=0.1,
    F14_active_loan_pressure=0.85)

HIGH_RISK = dict(
    F1_emi_to_income=0.88, F2_savings_drawdown=1.5, F3_salary_delay=14,
    F4_spend_shift=9.0, F5_auto_debit_fails=5, F6_lending_app_usage=6,
    F7_overdraft_freq=22, F8_stress_velocity=-0.75, F9_payment_entropy=12.0,
    F10_peer_stress=1.8, F12_cross_loan=1.2, F13_secondary_income=0.05,
    F14_active_loan_pressure=2.8)

GIG_RISK = dict(  # Moderate risk but gig worker uplift should tip it
    F1_emi_to_income=0.38, F2_savings_drawdown=0.3, F3_salary_delay=7,
    F4_spend_shift=2.5, F5_auto_debit_fails=1, F6_lending_app_usage=2,
    F7_overdraft_freq=6, F8_stress_velocity=-0.05, F9_payment_entropy=4.0,
    F10_peer_stress=0.8, F12_cross_loan=0.35, F13_secondary_income=0.4,
    F14_active_loan_pressure=0.72)

COLD_START = dict(  # Thin file, cold start, minimal history
    F1_emi_to_income=0.30, F2_savings_drawdown=0.0, F3_salary_delay=2,
    F4_spend_shift=1.8, F5_auto_debit_fails=0, F6_lending_app_usage=0,
    F7_overdraft_freq=1, F8_stress_velocity=0.0, F9_payment_entropy=2.0,
    F10_peer_stress=0.5, F12_cross_loan=2.5, F13_secondary_income=0.0,
    F14_active_loan_pressure=0.55)

VERY_HIGH = dict(  # Extreme stress — multiple compounding signals
    F1_emi_to_income=1.10, F2_savings_drawdown=2.0, F3_salary_delay=18,
    F4_spend_shift=12.0, F5_auto_debit_fails=7, F6_lending_app_usage=8,
    F7_overdraft_freq=28, F8_stress_velocity=-1.5, F9_payment_entropy=15.0,
    F10_peer_stress=2.2, F12_cross_loan=2.0, F13_secondary_income=0.0,
    F14_active_loan_pressure=4.5)

SAFE_GIG = dict(  # Low-risk profile but has gig flag — should show uplift but stay GREEN/MEDIUM
    F1_emi_to_income=0.08, F2_savings_drawdown=0.0, F3_salary_delay=1,
    F4_spend_shift=0.9, F5_auto_debit_fails=0, F6_lending_app_usage=0,
    F7_overdraft_freq=0, F8_stress_velocity=0.5, F9_payment_entropy=1.0,
    F10_peer_stress=0.3, F12_cross_loan=0.05, F13_secondary_income=0.5,
    F14_active_loan_pressure=0.15)

test_cases = [
    score_customer(LOW_RISK,   0, 0, "1. Low Risk (Salaried, stable)"),
    score_customer(MEDIUM_RISK,0, 0, "2. Medium Risk (Borderline)"),
    score_customer(HIGH_RISK,  0, 0, "3. High Risk (Multi-stress)"),
    score_customer(GIG_RISK,   1, 0, "4. Gig Worker (uplift applied)"),
    score_customer(COLD_START, 0, 1, "5. Cold Start / Thin File"),
    score_customer(VERY_HIGH,  0, 0, "6. Extreme Stress"),
    score_customer(SAFE_GIG,   1, 0, "7. Safe Gig (low-risk, gig uplift)"),
]

results_df = pd.DataFrame(test_cases)
print(results_df.to_string(index=False))

# ═════════════════════════════════════════════════════════════
# TEST 2: BANDING CORRECTNESS CHECK ON FULL SCORED DATASET
# ═════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("TEST 2: BANDING CORRECTNESS — FULL SCORED DATASET VERIFICATION")
print("=" * 70)

df_scored = pd.read_csv('modeltraining/anvaya_scored_dataset.csv')

green_wrongly_medium_or_high = df_scored[(df_scored['risk_band'] == 'GREEN') & (df_scored['PD_final'] > 5.0)]
medium_wrongly_green = df_scored[(df_scored['risk_band'] == 'MEDIUM') & (df_scored['PD_final'] <= 5.0)]
medium_wrongly_high  = df_scored[(df_scored['risk_band'] == 'MEDIUM') & (df_scored['PD_final'] >= 15.0)]
high_wrongly_medium  = df_scored[(df_scored['risk_band'] == 'HIGH')   & (df_scored['PD_final'] < 15.0)]

print(f"GREEN customers with PD > 5%   (contradiction): {len(green_wrongly_medium_or_high)}")
print(f"MEDIUM customers with PD <= 5% (should be GREEN): {len(medium_wrongly_green)}")
print(f"MEDIUM customers with PD >= 15% (should be HIGH): {len(medium_wrongly_high)}")
print(f"HIGH customers with PD < 15%   (should be MEDIUM): {len(high_wrongly_medium)}")

total_issues = len(green_wrongly_medium_or_high) + len(medium_wrongly_green) + len(medium_wrongly_high) + len(high_wrongly_medium)
if total_issues == 0:
    print("\n  BANDING IS 100% CONSISTENT — Zero contradictions found.")
else:
    print(f"\n  WARNING: {total_issues} banding contradictions found!")

# ═════════════════════════════════════════════════════════════
# TEST 3: MONOTONICITY CHECK — INCREMENTALLY WORSEN PROFILE
# ═════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("TEST 3: MONOTONICITY — INCREMENTALLY WORSENING LOW-RISK PROFILE")
print("=" * 70)

steps = [
    ("Step 0 — Baseline (clean)", dict(F1_emi_to_income=0.12, F2_savings_drawdown=0.0, F3_salary_delay=0, F4_spend_shift=1.0, F5_auto_debit_fails=0, F6_lending_app_usage=0, F7_overdraft_freq=0, F8_stress_velocity=0.3, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=0.2), 0, 0),
    ("Step 1 — EMI ratio 0.12→0.45", dict(F1_emi_to_income=0.45, F2_savings_drawdown=0.0, F3_salary_delay=0, F4_spend_shift=1.0, F5_auto_debit_fails=0, F6_lending_app_usage=0, F7_overdraft_freq=0, F8_stress_velocity=0.3, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=0.2), 0, 0),
    ("Step 2 — +4 day salary delay",  dict(F1_emi_to_income=0.45, F2_savings_drawdown=0.0, F3_salary_delay=4, F4_spend_shift=1.0, F5_auto_debit_fails=0, F6_lending_app_usage=0, F7_overdraft_freq=0, F8_stress_velocity=0.3, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=0.2), 0, 0),
    ("Step 3 — +10 overdraft days",   dict(F1_emi_to_income=0.45, F2_savings_drawdown=0.0, F3_salary_delay=4, F4_spend_shift=1.0, F5_auto_debit_fails=0, F6_lending_app_usage=0, F7_overdraft_freq=10, F8_stress_velocity=0.3, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=0.2), 0, 0),
    ("Step 4 — +3 auto-debit fails",  dict(F1_emi_to_income=0.45, F2_savings_drawdown=0.0, F3_salary_delay=4, F4_spend_shift=1.0, F5_auto_debit_fails=3, F6_lending_app_usage=0, F7_overdraft_freq=10, F8_stress_velocity=0.3, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=0.2), 0, 0),
    ("Step 5 — Loan pressure 0.2→2.0",dict(F1_emi_to_income=0.45, F2_savings_drawdown=0.0, F3_salary_delay=4, F4_spend_shift=1.0, F5_auto_debit_fails=3, F6_lending_app_usage=0, F7_overdraft_freq=10, F8_stress_velocity=0.3, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=2.0), 0, 0),
    ("Step 6 — Stress velocity -0.8", dict(F1_emi_to_income=0.45, F2_savings_drawdown=0.0, F3_salary_delay=4, F4_spend_shift=1.0, F5_auto_debit_fails=3, F6_lending_app_usage=0, F7_overdraft_freq=10, F8_stress_velocity=-0.8, F9_payment_entropy=0.5, F10_peer_stress=0.4, F12_cross_loan=0.1, F13_secondary_income=0.0, F14_active_loan_pressure=2.0), 0, 0),
]

mono_rows = []
prev_pd = -1
monotonic = True
for label, feats, gig, cold in steps:
    r = score_customer(feats, gig, cold, label)
    pd_val = r['PD_Final (%)']
    direction = "↑" if pd_val > prev_pd else ("→" if pd_val == prev_pd else "↓ WARNING")
    if pd_val < prev_pd and prev_pd >= 0:
        monotonic = False
    mono_rows.append({'Step': label, 'PD_Final (%)': pd_val, 'Band': r['Band'], 'Direction': direction})
    prev_pd = pd_val

mono_df = pd.DataFrame(mono_rows)
print(mono_df.to_string(index=False))
print(f"\n  Monotonicity: {'PASS — PD increases at every step.' if monotonic else 'FAIL — PD decreased at some step!'}")

# ═════════════════════════════════════════════════════════════
# TEST 4: PERFORMANCE SPOT-CHECK ON 5,000 SAMPLE
# ═════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("TEST 4: PERFORMANCE SPOT-CHECK (random 5,000 rows)")
print("=" * 70)

from sklearn.metrics import roc_auc_score, accuracy_score

sample = df_scored.sample(5000, random_state=99)
y_s    = sample['default_flag']
pd_s   = sample['PD_final'] / 100.0
pred_s = (pd_s >= 0.15).astype(int)

print(f"  AUC-ROC  : {roc_auc_score(y_s, pd_s):.4f}  (expected ~0.84)")
print(f"  Accuracy : {accuracy_score(y_s, pred_s)*100:.2f}%  (expected ~88%)")
print()
for band in ['GREEN', 'MEDIUM', 'HIGH']:
    sub = sample[sample['risk_band'] == band]
    if len(sub) > 0:
        print(f"  {band:8s}: {len(sub):>5,} customers  |  DR = {sub['default_flag'].mean()*100:.1f}%  "
              f"(expected: GREEN~2.6%, MEDIUM~8-9%, HIGH~40%+)")

print("\n  All done. See summary above.")
