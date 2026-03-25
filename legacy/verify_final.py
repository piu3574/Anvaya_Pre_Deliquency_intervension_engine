import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os, pickle

df   = pd.read_csv('modeltraining/anvaya_scored_dataset.csv')
y    = df['default_flag']
pd_f = df['PD_final'] / 100.0

print('=== INDEPENDENT METRIC VERIFICATION FROM CSV (SINGLE MODEL) ===')
print(f'Total rows       : {len(df):,}')
print(f'Overall DR       : {y.mean()*100:.2f}%')
print(f'AUC-ROC          : {roc_auc_score(y, pd_f):.4f}')
print(f'Accuracy@15pct   : {accuracy_score(y, (pd_f >= 0.15).astype(int))*100:.2f}%')

with open('modeltraining/banding_config.pkl', 'rb') as f:
    conf = pickle.load(f)
g_thresh = conf['green']
h_thresh = conf['high']

print()
print(f'--- Band Distribution (Design B - PD_final only | G<={g_thresh}%, H>={h_thresh}%) ---')
green  = df[df['PD_final'] <= g_thresh]
medium = df[(df['PD_final'] > g_thresh) & (df['PD_final'] < h_thresh)]
high   = df[df['PD_final'] >= h_thresh]
for name, sub in [('GREEN', green), ('MEDIUM', medium), ('HIGH', high)]:
    if len(sub) > 0:
        print(f'  {name:10s}: {len(sub):>6,} customers | DR = {sub["default_flag"].mean()*100:.1f}%')

print()
# Alignment check: risk_band column vs PD_final thresholds
is_green_ok  = ((df['risk_band'] == 'GREEN')  == (df['PD_final'] <= g_thresh))
is_medium_ok = ((df['risk_band'] == 'MEDIUM') == ((df['PD_final'] > g_thresh) & (df['PD_final'] < h_thresh)))
is_high_ok   = ((df['risk_band'] == 'HIGH')   == (df['PD_final'] >= h_thresh))
all_ok = (is_green_ok & is_medium_ok & is_high_ok).all()
print(f'risk_band column 100% aligned with PD_final thresholds: {all_ok}')
contradictions = (~(is_green_ok & is_medium_ok & is_high_ok)).sum()
print(f'Contradictions   : {contradictions}')

print()
print('--- Artifact Checks ---')
for path in ['modeltraining/lgbm_calibrated.pkl', 'modeltraining/woe_lookup.pkl', 'explainability/shap_explanations.jsonl']:
    exists = os.path.exists(path)
    print(f'  [{"OK" if exists else "MISSING"}] {path}')

print()
print('--- Final Repo Summary ---')
for folder, fnames in [
    ('root',           ['generate_dataset.py','pipeline_tests.py','compute_thresholds.py','README.md']),
    ('modeltraining/', ['train_xgboost.py','train_lightgbm.py','shap_explainer.py','xgb_fastpath.pkl',
                        'lgbm_raw.pkl','lgbm_calibrated.pkl','woe_lookup.pkl','iv_ranking.pkl',
                        'anvaya_scored_dataset.csv','features_with_xgb.pkl']),
    ('data/',          ['anvaya_scored_dataset_vFINAL.csv']),
    ('explainability/',['shap_explanations.jsonl','shap_baseline.pkl']),
    ('dataset/',       ['barclays_bank_synthetic_data.csv']),
    ('docs/',          ['ml_architecture_plan.md','prd.md']),
    ('api/',           ['main.py']),
]:
    for fn in fnames:
        path = fn if folder == 'root' else os.path.join(folder, fn)
        exists = os.path.exists(path)
        size   = f'{os.path.getsize(path)/1e6:.1f} MB' if exists else 'MISSING'
        mark   = 'OK' if exists else 'MISSING'
        print(f'  [{mark}] {folder}{fn:45s} {size}')
