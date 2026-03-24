"""
shap_explainer.py — Authoritative Single-Model SHAP Stage
Runs after single-model PD training.
- Loads calibrated LightGBM and scored dataset.
- Computes baseline PD from background sample.
- Generates SHAP explanations for MEDIUM and HIGH risk customers.
"""
import pandas as pd
import numpy as np
import joblib
import shap
import json
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

# ── 1. LOAD ARTIFACTS ───────────────────────────────────────────────────────
print("1. Loading models and scored dataset...")
lgb_raw  = joblib.load('modeltraining/lgbm_raw.pkl')
lgb_cal  = joblib.load('modeltraining/lgbm_calibrated.pkl')
df       = pd.read_csv('modeltraining/anvaya_scored_dataset.csv')

with open('modeltraining/woe_lookup.pkl', 'rb') as f:
    woe_lookup = pickle.load(f)

with open('modeltraining/banding_config.pkl', 'rb') as f:
    banding_config = pickle.load(f)

ALL_WOE = [f'{feat}_WoE' for feat in woe_lookup.keys()]
X_all = df[ALL_WOE].fillna(0)

print(f"   Dataset: {len(df):,} rows  |  MEDIUM+HIGH: {(df['risk_band']!='GREEN').sum():,}")

# ── 2. BACKGROUND & BASELINE ────────────────────────────────────────────────
print("\n2. Computing baseline PD from 500-row stratified background sample...")
# Stratified sample for stable baseline
background = df.groupby('default_flag', group_keys=False).apply(
    lambda g: g.sample(min(len(g), 250), random_state=42)
).sample(500, random_state=42)

X_bg = background[ALL_WOE].fillna(0)
bg_preds = lgb_raw.predict_proba(X_bg)[:, 1]
baseline_pd_lgb = float(bg_preds.mean())
print(f"   Baseline PD (LGB, portfolio average): {baseline_pd_lgb*100:.2f}%")

# Save baseline artifact
os.makedirs('explainability', exist_ok=True)
with open('explainability/shap_baseline.pkl', 'wb') as f:
    pickle.dump({
        'baseline_pd_lgb': baseline_pd_lgb, 
        'baseline_version': 'S1-2026-Q1',
        'background_size': 500
    }, f)

# ── 3. TREE EXPLAINER ───────────────────────────────────────────────────────
print("\n3. Building SHAP TreeExplainer...")
# Using interventional mode for consistency across various LGB versions/wrappers
explainer = shap.TreeExplainer(
    lgb_raw,
    data=X_bg.values,
    feature_perturbation='interventional',
    model_output='raw'
)

# ── 4. LABEL MAPPING ────────────────────────────────────────────────────────
LABEL_MAP = {
    'F1_emi_to_income_WoE':      ('EMI burden',            'Monthly loan repayments are high relative to income.'),
    'F2_savings_drawdown_WoE':   ('Savings depletion',     'Savings balance has dropped sharply over last 60–90 days.'),
    'F3_salary_delay_WoE':       ('Salary arriving late',  'Salary credited later than expected relative to EMI due date.'),
    'F4_spend_shift_WoE':        ('Spending shift',        'Discretionary spending pattern has shifted materially.'),
    'F5_auto_debit_fails_WoE':   ('Payment failures',      'Auto-debits failed due to insufficient funds recently.'),
    'F6_lending_app_usage_WoE':  ('Credit-seeking signals','Customer has been querying or using lending apps recently.'),
    'F7_overdraft_freq_WoE':     ('Overdraft days',        'Account went into overdraft on multiple days last month.'),
    'F8_stress_velocity_WoE':    ('Stress increasing',     'Balance trajectory is declining month-on-month.'),
    'F9_payment_entropy_WoE':    ('Irregular payments',    'EMI payment dates are inconsistent across months.'),
    'F10_peer_stress_WoE':       ('Peer group under pressure', 'Customers with similar employment profile are showing stress.'),
    'F12_cross_loan_WoE':        ('Loan-to-tenure risk',   'High number of active loans relative to account age.'),
    'F13_secondary_income_WoE':  ('Limited income diversity','Customer lacks supplementary income sources.'),
    'F14_active_loan_pressure_WoE':('High loan utilisation','Outstanding loan balance is large relative to credit limit.'),
}

def build_shap_payload(row, shap_vals, baseline_p, pd_actual):
    all_features = []
    for feat, sv in zip(ALL_WOE, shap_vals):
        all_features.append({
            'feature': feat, 
            'woe_value': round(float(row[feat]), 4),
            'shap_value': round(float(sv), 6)
        })

    sorted_feats = sorted(all_features, key=lambda x: abs(x['shap_value']), reverse=True)
    
    # Top 5 positive drivers (risk factors)
    top_drivers = []
    for f in sorted_feats:
        if f['shap_value'] > 0.002 and len(top_drivers) < 5:
            label, detail = LABEL_MAP.get(f['feature'], (f['feature'], ''))
            top_drivers.append({
                'feature': f['feature'],
                'shap_value': f['shap_value'],
                'plain_language_label': label,
                'plain_language_detail': detail
            })

    # Top negative driver (mitigating factor)
    mitigating = []
    neg_feats = [f for f in sorted_feats if f['shap_value'] < -0.002]
    if neg_feats:
        f = neg_feats[0]
        label, detail = LABEL_MAP.get(f['feature'], (f['feature'], ''))
        mitigating.append({
            'feature': f['feature'], 'shap_value': f['shap_value'],
            'plain_language_label': label, 'plain_language_detail': detail
        })

    return {
        'customer_id': str(row.get('customer_id', 'Unknown')),
        'pd_final': round(float(row['PD_final']), 4),
        'risk_band': str(row['risk_band']),
        'baseline_pd': round(baseline_p * 100, 4),
        'top_drivers': top_drivers,
        'mitigating_factors': mitigating,
        'uplifts': {
            'gig_worker': bool(row.get('gig_worker_flag', 0)),
            'cold_start': bool(row.get('cold_start_flag', 0))
        }
    }

# ── 5. RUN SHAP ─────────────────────────────────────────────────────────────
print("\n4. Generating SHAP for non-GREEN customers...")
df_explain = df[df['risk_band'] != 'GREEN'].copy().reset_index(drop=True)
X_explain  = df_explain[ALL_WOE].fillna(0)

# Batch compute SHAP in logit space
shap_values_logit = explainer.shap_values(X_explain)
baseline_logit = float(explainer.expected_value)

# Convert to probability space (linear approximation at baseline)
def sigmoid(x): return 1 / (1 + np.exp(-x))
baseline_prob = sigmoid(baseline_logit)
sigma_prime   = baseline_prob * (1 - baseline_prob)
shap_values_prob = shap_values_logit * sigma_prime

# ── 6. SAVE OUTPUT ──────────────────────────────────────────────────────────
out_path = 'explainability/shap_explanations.jsonl'
print(f"5. Saving to {out_path}...")
with open(out_path, 'w') as f:
    for i, (_, row) in enumerate(df_explain.iterrows()):
        payload = build_shap_payload(row, shap_values_prob[i], baseline_prob, row['PD_final']/100)
        f.write(json.dumps(payload) + '\n')

# ── 7. EXAMPLES ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SAMPLE EXPLANATIONS (Single-Model)")
print("="*60)
if len(df_explain) > 0:
    sample_size = min(3, len(df_explain))
    samples = df_explain.sample(sample_size, random_state=42).index
    for idx in samples:
        row = df_explain.iloc[idx]
        p = build_shap_payload(row, shap_values_prob[idx], baseline_prob, 0)
        print(f"\nID: {p['customer_id']} | PD: {p['pd_final']:.1f}% | Band: {p['risk_band']}")
        print(f"Top Drivers: {[d['plain_language_label'] for d in p['top_drivers']]}")

print("\n--- SHAP STAGE COMPLETE ---")
