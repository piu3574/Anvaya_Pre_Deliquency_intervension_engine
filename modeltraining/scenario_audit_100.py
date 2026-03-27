import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from scipy.stats import ks_2samp
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. LOAD MODELS
m_xgb = joblib.load('xgb_model.pkl')
m_lgbm = joblib.load('lgbm_model.pkl')
m_meta = joblib.load('ensemble_meta.pkl')
m_scaler = joblib.load('meta_scaler.pkl')
bc = pickle.load(open('banding_config.pkl', 'rb'))
t_green = bc['green']/100.0
t_red = bc['red']/100.0

# 2. LOAD DATA
df_raw = pd.read_csv('../dataset/hyper_realistic_portfolio_100k.csv')
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
X.fillna(0, inplace=True)

# EXACT 70/30 Split used during Training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_test = X_test[m_xgb.get_booster().feature_names]

p_xgb = m_xgb.predict_proba(X_test)[:,1]
p_lgbm = m_lgbm.predict_proba(X_test)[:,1]
X_meta = m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm}))
p_final = m_meta.predict_proba(X_meta)[:,1]

y_pred = (p_final >= t_red).astype(int)

auc_val = roc_auc_score(y_test, p_final)
gini = 2 * auc_val - 1
ks = ks_2samp(p_final[y_test==1], p_final[y_test==0]).statistic

print("=== FINAL METRICS (30% TEST SET) ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC  : {auc_val:.4f}")
print(f"Gini     : {gini:.4f}")
print(f"KS Stat  : {ks:.4f}")
print(f"Test Set Default Rate: {y_test.mean()*100:.2f}%")

df_test = pd.DataFrame({'p': p_final, 'y': y_test})
df_test['band'] = np.where(df_test['p'] >= t_red, 'RED', np.where(df_test['p'] >= t_green, 'YELLOW', 'GREEN'))

print("\n=== BAND DISTRIBUTION (TEST SET) ===")
for b in ['GREEN', 'YELLOW', 'RED']:
    sub = df_test[df_test['band'] == b]
    dr = sub['y'].mean() * 100 if len(sub) > 0 else 0
    pct = len(sub) / len(df_test) * 100
    print(f"{b:6s} | {pct:5.1f}% of users | Default Rate: {dr:5.1f}%")

# 3. 100 DIVERSE SCENARIOS
print("\n=== GENERATING 100 DIVERSE SCENARIOS ===")
np.random.seed(99)
scenarios = []
labels = []
# 25 Safe prime (Low DTI, no bounces, old vintage)
for _ in range(25):
    scenarios.append([np.random.uniform(0, 0.2), 0, 0, 0, 0, np.random.uniform(0, 0.3), np.random.uniform(0.5, 1), np.random.uniform(0.3, 1), np.random.uniform(0, 0.1), 0])
    labels.append("Prime Safe")
# 25 High DTI but clean history
for _ in range(25):
    scenarios.append([np.random.uniform(0.6, 0.9), 0, 0, 0, 0, np.random.uniform(0.5, 0.9), np.random.uniform(0.1, 1), np.random.uniform(0.3, 1), np.random.uniform(0, 0.1), 0])
    labels.append("High DTI Safe")
# 25 Recent Bouncers (stress_f5 high)
for _ in range(25):
    scenarios.append([np.random.uniform(0.3, 0.9), np.random.uniform(0, 0.5), 0, np.random.uniform(0.4, 1), np.random.uniform(0, 0.3), np.random.uniform(0.3, 0.9), np.random.uniform(0, 0.5), np.random.uniform(0.2, 0.8), np.random.uniform(0.1, 0.5), 0])
    labels.append("Recent Bouncers")
# 25 Disaster / Overdraft & Lending Apps
for _ in range(25):
    scenarios.append([np.random.uniform(0.5, 1), np.random.uniform(0.5, 1), np.random.uniform(0.5, 1), np.random.uniform(0.4, 1), np.random.uniform(0.5, 1), np.random.uniform(0.6, 1), np.random.uniform(0, 0.3), np.random.uniform(0.2, 0.8), np.random.uniform(0.5, 1), np.random.uniform(0.5, 1)])
    labels.append("Disaster")

X_100 = pd.DataFrame(scenarios, columns=m_xgb.get_booster().feature_names)
p100_xgb = m_xgb.predict_proba(X_100)[:,1]
p100_lgbm = m_lgbm.predict_proba(X_100)[:,1]
p100_final = m_meta.predict_proba(m_scaler.transform(pd.DataFrame({'xgb': p100_xgb, 'lgbm': p100_lgbm})))[:,1]

df_100 = pd.DataFrame({'Scenario': labels, 'p': p100_final})
df_100['band'] = np.where(df_100['p'] >= t_red, 'RED', np.where(df_100['p'] >= t_green, 'YELLOW', 'GREEN'))

print("\n=== 100 SCENARIO RESULTS ===")
for grp, sub in df_100.groupby('Scenario'):
    print(f"Group: {grp:15s} | Mean PD: {sub['p'].mean():.4f} | Bands: {sub['band'].value_counts().to_dict()}")
