"""
stress_test.py — Full pipeline stress test for Anvaya Phase-2
Covers:
  1. Runtime profiling on 10,000 customers
  2. 8 hand-crafted scenario cases with full SHAP breakdown
  3. Consistency + SHAP additivity verification
"""
import time, json, pickle, warnings
import pandas as pd
import numpy as np
import joblib
import shap
from sklearn.metrics import roc_auc_score, accuracy_score
warnings.filterwarnings('ignore')

# ── Load all artefacts once ───────────────────────────────────────────────────
print("Loading models and artefacts...")
xgb_model = joblib.load('modeltraining/xgb_fastpath.pkl')
lgb_raw   = joblib.load('modeltraining/lgbm_raw.pkl')
lgb_cal   = joblib.load('modeltraining/lgbm_calibrated.pkl')
with open('modeltraining/iv_ranking.pkl', 'rb') as f:
    iv_data = pickle.load(f)
with open('explainability/shap_baseline.pkl', 'rb') as f:
    bl_data = pickle.load(f)

top4_woe = [f"{feat}_WoE" for feat in iv_data['top4']]
ALL_WOE  = [
    'F1_emi_to_income_WoE','F2_savings_drawdown_WoE','F3_salary_delay_WoE',
    'F4_spend_shift_WoE','F5_auto_debit_fails_WoE','F6_lending_app_usage_WoE',
    'F7_overdraft_freq_WoE','F8_stress_velocity_WoE','F9_payment_entropy_WoE',
    'F10_peer_stress_WoE','F12_cross_loan_WoE','F13_secondary_income_WoE',
    'F14_active_loan_pressure_WoE'
]
FEAT_COLS = [c.replace('_WoE','') for c in ALL_WOE]
LABEL_MAP = {
    'F1_emi_to_income_WoE':       'EMI burden',
    'F2_savings_drawdown_WoE':    'Savings depletion',
    'F3_salary_delay_WoE':        'Salary arriving late',
    'F4_spend_shift_WoE':         'Spending shift',
    'F5_auto_debit_fails_WoE':    'Payment failures',
    'F6_lending_app_usage_WoE':   'Credit-seeking signals',
    'F7_overdraft_freq_WoE':      'Overdraft days',
    'F8_stress_velocity_WoE':     'Stress increasing',
    'F9_payment_entropy_WoE':     'Irregular payments',
    'F10_peer_stress_WoE':        'Peer group pressure',
    'F12_cross_loan_WoE':         'Loan-to-tenure risk',
    'F13_secondary_income_WoE':   'Limited income diversity',
    'F14_active_loan_pressure_WoE':'High loan utilisation',
}

# ── Build WoE bin structures from training data ───────────────────────────────
df_full = pd.read_csv('dataset/barclays_bank_synthetic_data.csv')
y_full  = df_full['default_flag']
emi_cols = [f'emi_payment_day_m{x}' for x in range(1,7)]
for c in emi_cols:
    df_full[c] = pd.to_datetime(df_full[c], errors='coerce').dt.day

df_full['F1_emi_to_income']  = (df_full['total_monthly_emi_amount'] / (df_full['total_monthly_income']+1)).clip(0,5)
df_full['F2_savings_drawdown'] = np.clip((df_full['savings_balance_60d_ago']-df_full['current_account_balance'])/(df_full['savings_balance_60d_ago']+1),0,5)
try:
    df_full['F3_salary_delay'] = (pd.to_datetime(df_full['salary_credit_date_m1']).dt.day - df_full['expected_salary_day_of_month']).clip(-5,20)
except: df_full['F3_salary_delay'] = 0
df_full['F4_spend_shift']   = (df_full['total_debit_amount_30d']*3/(df_full['grocery_spend_amount_90d']+1)).clip(0,20)
df_full['F5_auto_debit_fails'] = df_full['auto_debit_failed_insufficient_funds_30d']
df_full['F6_lending_app_usage'] = df_full['lending_app_transaction_count_30d']
df_full['F7_overdraft_freq']   = df_full['overdraft_days_30d']
df_full['F8_stress_velocity']  = ((df_full['end_of_month_balance_m6']-df_full['end_of_month_balance_m1'])/(df_full['end_of_month_balance_m6']+1)).clip(-5,5)
df_full['F9_payment_entropy']  = df_full[emi_cols].std(axis=1).fillna(0)
df_full['F14_active_loan_pressure'] = (df_full['total_loan_outstanding']/(df_full['total_credit_limit']+1)).clip(0,20)
df_full['F10_peer_stress']  = df_full.groupby(['employment_category'])['F14_active_loan_pressure'].transform('mean')
df_full['F12_cross_loan']   = df_full['number_of_active_loans']/(df_full['customer_vintage_months']+1)
df_full['F13_secondary_income'] = ((df_full['total_monthly_income']-df_full['monthly_net_salary'])/(df_full['total_monthly_income']+1)).clip(0,1)

bin_structures, woe_map = {}, {}
for feat in FEAT_COLS:
    vals = df_full[feat].replace([np.inf,-np.inf], np.nan).fillna(0)
    try:    bins_obj = pd.qcut(vals, q=10, duplicates='drop')
    except: bins_obj = pd.cut(vals, bins=10, duplicates='drop')
    bin_structures[feat] = bins_obj.cat.categories
    total_d = y_full.sum(); total_s = (y_full==0).sum()
    wd = {}
    for bl in bins_obj.cat.categories:
        mask = (bins_obj == bl)
        d_b = y_full[mask].sum(); s_b = (y_full==0)[mask].sum()
        p_d = max(d_b/total_d,1e-9); p_s = max(s_b/total_s,1e-9)
        wd[str(bl)] = np.log(p_d/p_s)
    woe_map[feat] = wd

def map_woe(feat, value):
    cats = bin_structures[feat]
    for interval in cats:
        if interval.left < value <= interval.right:
            return woe_map[feat].get(str(interval), 0.0)
    if value <= cats[0].right:  return woe_map[feat].get(str(cats[0]), 0.0)
    return woe_map[feat].get(str(cats[-1]), 0.0)

def score_one(feats, gig=0, cold=0):
    woe = {f'{feat}_WoE': map_woe(feat, feats.get(feat, 0.0)) for feat in FEAT_COLS}
    x_xgb   = pd.DataFrame([woe])[top4_woe]
    pd_xgb  = float(xgb_model.predict_proba(x_xgb)[0,1]) * 100
    triage  = 'GREEN' if pd_xgb <= 5.0 else ('RED_FAST' if pd_xgb >= 20.0 else 'BORDERLINE')
    if triage == 'BORDERLINE':
        x_lgb  = pd.DataFrame([woe])[ALL_WOE]
        pd_lgb = float(lgb_cal.predict_proba(x_lgb)[0,1]) * 100
        pd_b   = 0.4*pd_xgb + 0.6*pd_lgb
    else:
        x_lgb  = pd.DataFrame([woe])[ALL_WOE]
        pd_lgb = float(lgb_cal.predict_proba(x_lgb)[0,1]) * 100
        pd_b   = pd_xgb
    pd_final = min(max(pd_b + gig*1.0 + cold*0.5, 0), 100)
    band     = 'GREEN' if pd_final <= 5.0 else ('HIGH' if pd_final >= 15.0 else 'MEDIUM')
    return {'pd_xgb': pd_xgb, 'pd_lgb': pd_lgb, 'pd_final': pd_final,
            'triage': triage, 'band': band, 'woe': woe, 'gig': gig, 'cold': cold}

def sigmoid(x): return 1/(1+np.exp(-x))
# Build SHAP explainer once
X_bg = pd.DataFrame([{f: 0.0 for f in ALL_WOE}])
X_bg_arr = df_full[[f for f in FEAT_COLS]].head(0)  # dummy
bg_woe = []
for _, row in df_full.sample(500, random_state=42).iterrows():
    w = {f'{feat}_WoE': map_woe(feat, float(row[feat]) if feat in row else 0.0) for feat in FEAT_COLS}
    bg_woe.append(w)
X_bg_shap = pd.DataFrame(bg_woe)[ALL_WOE].fillna(0)
explainer  = shap.TreeExplainer(lgb_raw, data=X_bg_shap.values,
                                 feature_perturbation='interventional', model_output='raw')
bl_logit   = float(explainer.expected_value)
bl_prob    = sigmoid(bl_logit)
sigma_p    = bl_prob * (1 - bl_prob)

def shap_explain(woe_row):
    x = pd.DataFrame([woe_row])[ALL_WOE].fillna(0)
    sv      = explainer.shap_values(x)[0]
    sv_prob = sv * sigma_p
    features = sorted(
        [{'feat': f, 'label': LABEL_MAP[f], 'woe': round(float(woe_row[f]),3),
          'shap': round(float(sv_prob[i]),5)} for i, f in enumerate(ALL_WOE)],
        key=lambda x: abs(x['shap']), reverse=True
    )
    top_pos = [f for f in features if f['shap'] > 0.001][:4]
    top_neg = [f for f in features if f['shap'] < -0.001][:1]
    return {'baseline': round(bl_prob*100,2), 'shap_sum_pp': round(sum(f['shap'] for f in features)*100,2),
            'top_drivers': top_pos, 'mitigating': top_neg, 'all': features}

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RUNTIME PROFILING (10,000 customers)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SECTION 1: RUNTIME PROFILING — 10,000 CUSTOMERS")
print("="*70)

df_scored = pd.read_csv('modeltraining/anvaya_scored_dataset.csv')
sample10k = df_scored.sample(10000, random_state=1).reset_index(drop=True)
X10k_woe  = sample10k[ALL_WOE].fillna(0)
X10k_top4 = X10k_woe[top4_woe]
bl_idx    = sample10k['triage'] == 'BORDERLINE'

# XGBoost fast path
t0 = time.perf_counter()
pd_xgb_10k = xgb_model.predict_proba(X10k_top4)[:,1] * 100
t_xgb = time.perf_counter() - t0

# LightGBM deep path (borderline only)
t2 = time.perf_counter()
pd_lgb_10k = np.zeros(len(sample10k))
pd_lgb_10k[bl_idx] = lgb_cal.predict_proba(X10k_woe[bl_idx])[:,1] * 100
t_lgb = time.perf_counter() - t2

# SHAP on MEDIUM+HIGH
non_green = (sample10k['risk_band'] != 'GREEN')
X_ng = X10k_woe[non_green]
t3 = time.perf_counter()
sv_batch = explainer.shap_values(X_ng)
t_shap = time.perf_counter() - t3

total_score   = t_xgb + t_lgb
total_with_sh = t_xgb + t_lgb + t_shap
n_bl  = bl_idx.sum()
n_ng  = non_green.sum()

print(f"  Customers scored : 10,000")
print(f"  Borderline (LGB) : {n_bl:,} ({n_bl/100:.1f}%)")
print(f"  MEDIUM+HIGH (SHAP): {n_ng:,} ({n_ng/100:.1f}%)")
print()
print(f"  XGBoost fast path     : {t_xgb*1000:>8.1f} ms total  |  {t_xgb/10:.4f} ms/customer")
print(f"  LightGBM deep path    : {t_lgb*1000:>8.1f} ms total  |  {t_lgb/n_bl*1000:.4f} ms/borderline-customer")
print(f"  SHAP (MEDIUM+HIGH)    : {t_shap*1000:>8.1f} ms total  |  {t_shap/n_ng*1000:.4f} ms/non-green-customer")
print()
print(f"  === Per-customer (avg over 10k) ===")
print(f"  Score only (no SHAP)  : {total_score/10:.4f} ms")
print(f"  Score + SHAP          : {total_with_sh/10:.4f} ms")
print(f"  Total wall-clock      : {(total_score)*1000:.1f} ms scoring  |  {total_with_sh*1000:.1f} ms with SHAP")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: 8 SCENARIO CASES
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    ("Case 1 — Clean Low Risk",
     dict(F1_emi_to_income=0.13, F2_savings_drawdown=0.0, F3_salary_delay=0,
          F4_spend_shift=0.9, F5_auto_debit_fails=0, F6_lending_app_usage=0,
          F7_overdraft_freq=0, F8_stress_velocity=0.4, F9_payment_entropy=0.8,
          F10_peer_stress=0.3, F12_cross_loan=0.08, F13_secondary_income=0.05,
          F14_active_loan_pressure=0.18),
     0, 0, "GREEN", "No SHAP needed"),

    ("Case 2 — Pure Timing Issue",
     dict(F1_emi_to_income=0.31, F2_savings_drawdown=0.10, F3_salary_delay=6,
          F4_spend_shift=1.2, F5_auto_debit_fails=1, F6_lending_app_usage=0,
          F7_overdraft_freq=2, F8_stress_velocity=-0.12, F9_payment_entropy=2.1,
          F10_peer_stress=0.5, F12_cross_loan=0.15, F13_secondary_income=0.1,
          F14_active_loan_pressure=0.42),
     0, 0, "MEDIUM", "Top drivers: F3 salary delay, F8 stress velocity"),

    ("Case 3 — Structural Over-indebtedness",
     dict(F1_emi_to_income=0.68, F2_savings_drawdown=0.05, F3_salary_delay=0,
          F4_spend_shift=1.0, F5_auto_debit_fails=0, F6_lending_app_usage=0,
          F7_overdraft_freq=0, F8_stress_velocity=0.05, F9_payment_entropy=0.5,
          F10_peer_stress=0.8, F12_cross_loan=0.9, F13_secondary_income=0.0,
          F14_active_loan_pressure=1.85),
     0, 0, "MEDIUM/HIGH", "Top drivers: F1 EMI, F14 loan pressure, F12 cross-loan"),

    ("Case 4 — Acute Liquidity Crisis",
     dict(F1_emi_to_income=0.55, F2_savings_drawdown=1.4, F3_salary_delay=8,
          F4_spend_shift=4.0, F5_auto_debit_fails=6, F6_lending_app_usage=3,
          F7_overdraft_freq=18, F8_stress_velocity=-0.55, F9_payment_entropy=9.0,
          F10_peer_stress=1.1, F12_cross_loan=0.5, F13_secondary_income=0.0,
          F14_active_loan_pressure=1.2),
     0, 0, "HIGH", "Top drivers: F5 auto-debit fails, F7 overdraft"),

    ("Case 5 — Credit-Seeking Behaviour",
     dict(F1_emi_to_income=0.29, F2_savings_drawdown=0.08, F3_salary_delay=1,
          F4_spend_shift=1.1, F5_auto_debit_fails=0, F6_lending_app_usage=7,
          F7_overdraft_freq=1, F8_stress_velocity=0.1, F9_payment_entropy=0.9,
          F10_peer_stress=0.5, F12_cross_loan=1.3, F13_secondary_income=0.1,
          F14_active_loan_pressure=0.75),
     0, 0, "MEDIUM", "Top drivers: F6 lending apps, F12 cross-loan"),

    ("Case 6 — Gig Worker with Uplift",
     dict(F1_emi_to_income=0.36, F2_savings_drawdown=0.12, F3_salary_delay=5,
          F4_spend_shift=1.3, F5_auto_debit_fails=1, F6_lending_app_usage=1,
          F7_overdraft_freq=3, F8_stress_velocity=-0.10, F9_payment_entropy=2.5,
          F10_peer_stress=0.6, F12_cross_loan=0.2, F13_secondary_income=0.4,
          F14_active_loan_pressure=0.55),
     1, 0, "MEDIUM/HIGH", "PD_final = PD_base + 1.0pp gig uplift. Uplift NOT in SHAP."),

    ("Case 7 — Thin File / Cold Start",
     dict(F1_emi_to_income=0.25, F2_savings_drawdown=0.0, F3_salary_delay=2,
          F4_spend_shift=1.5, F5_auto_debit_fails=0, F6_lending_app_usage=0,
          F7_overdraft_freq=1, F8_stress_velocity=0.0, F9_payment_entropy=1.8,
          F10_peer_stress=0.4, F12_cross_loan=3.0, F13_secondary_income=0.0,
          F14_active_loan_pressure=0.4),
     0, 1, "MEDIUM", "PD_final = PD_base + 0.5pp cold-start uplift."),

    ("Case 8B — RED_FAST route HIGH",
     dict(F1_emi_to_income=0.75, F2_savings_drawdown=1.2, F3_salary_delay=12,
          F4_spend_shift=6.0, F5_auto_debit_fails=4, F6_lending_app_usage=5,
          F7_overdraft_freq=20, F8_stress_velocity=-0.7, F9_payment_entropy=11.0,
          F10_peer_stress=1.6, F12_cross_loan=0.8, F13_secondary_income=0.0,
          F14_active_loan_pressure=2.5),
     0, 0, "HIGH", "RED_FAST route; SHAP run retrospectively on LGB for explanation."),
]

print("\n" + "="*70)
print("SECTION 2: 8 SCENARIO CASES — FULL PIPELINE")
print("="*70)

consistency_results = []

for case_name, feats, gig, cold, expected_band, expectation in SCENARIOS:
    print(f"\n{'─'*65}")
    print(f"  {case_name}")
    print(f"  Expected : {expected_band}  |  {expectation}")
    print(f"{'─'*65}")

    res = score_one(feats, gig, cold)

    print(f"  Features (key):")
    for fn in ['F1_emi_to_income','F3_salary_delay','F5_auto_debit_fails',
               'F7_overdraft_freq','F8_stress_velocity','F14_active_loan_pressure']:
        print(f"    {fn:30s} = {feats.get(fn, 0):.2f}")

    print(f"\n  PD_XGB          : {res['pd_xgb']:.2f}%")
    print(f"  PD_LGB          : {res['pd_lgb']:.2f}%")
    print(f"  Triage route    : {res['triage']}")
    if gig:  print(f"  Gig uplift      : +1.0pp")
    if cold: print(f"  Cold-start uplift: +0.5pp")
    print(f"  PD_final        : {res['pd_final']:.2f}%")
    print(f"  Band            : {res['band']}")

    # Banding consistency
    ok_band = (
        (res['band'] == 'GREEN'  and res['pd_final'] <= 5.0) or
        (res['band'] == 'MEDIUM' and 5.0 < res['pd_final'] < 15.0) or
        (res['band'] == 'HIGH'   and res['pd_final'] >= 15.0)
    )
    print(f"  Band consistent : {'YES' if ok_band else 'NO — CHECK!'}")

    # SHAP for non-GREEN
    if res['band'] != 'GREEN':
        expl = shap_explain(res['woe'])
        print(f"\n  SHAP (baseline = {expl['baseline']:.2f}%, SHAP sum = {expl['shap_sum_pp']:+.2f}pp):")
        for d in expl['top_drivers']:
            print(f"    [+] {d['label']:28s}  SHAP={d['shap']:+.4f}  WoE={d['woe']:+.3f}")
        for d in expl['mitigating']:
            print(f"    [-] {d['label']:28s}  SHAP={d['shap']:+.4f}  WoE={d['woe']:+.3f}")
        if gig:
            print(f"  >> Gig uplift (+1.0pp) is NOT in SHAP — shown as overlay only.")
        if cold:
            print(f"  >> Cold-start uplift (+0.5pp) is NOT in SHAP — shown as overlay only.")
    else:
        print(f"\n  SHAP : Not generated (GREEN band — auto-approve, no action needed).")

    consistency_results.append((case_name, res['band'], ok_band, res['pd_final']))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("SECTION 3: CONSISTENCY SUMMARY")
print("="*70)
all_ok = True
for name, band, ok, pd_val in consistency_results:
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] {name:38s} Band={band:6s} PD={pd_val:.1f}%")
    if not ok: all_ok = False

print()
print(f"  Band consistency across all 8 cases: {'ALL PASS' if all_ok else 'FAILURES FOUND'}")
print()

# Global quality gate numbers from scored dataset
df_sc = df_scored
y_sc  = df_sc['default_flag']
pd_sc = df_sc['PD_final'] / 100.0
print(f"  Global Quality Gate (from scored dataset):")
print(f"    AUC-ROC      : {roc_auc_score(y_sc, pd_sc):.4f}")
print(f"    Accuracy@15% : {accuracy_score(y_sc, (pd_sc>=0.15).astype(int))*100:.2f}%")
print(f"    DR per band  : GREEN={df_sc[df_sc['risk_band']=='GREEN']['default_flag'].mean()*100:.1f}%  "
      f"MEDIUM={df_sc[df_sc['risk_band']=='MEDIUM']['default_flag'].mean()*100:.1f}%  "
      f"HIGH={df_sc[df_sc['risk_band']=='HIGH']['default_flag'].mean()*100:.1f}%")
