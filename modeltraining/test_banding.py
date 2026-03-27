import pandas as pd
import numpy as np
import joblib
import pickle
import os

# Load models
m_xgb = joblib.load('xgb_model.pkl')
m_lgbm = joblib.load('lgbm_model.pkl')
m_meta = joblib.load('ensemble_meta.pkl')
m_scaler = joblib.load('meta_scaler.pkl')
bc = pickle.load(open('banding_config.pkl', 'rb'))

t_green = bc['green'] / 100.0
t_red = bc['red'] / 100.0

print(f"=== BANDING THRESHOLDS ===")
print(f"GREEN : PD < {t_green:.4f}")
print(f"YELLOW: {t_green:.4f} <= PD < {t_red:.4f}")
print(f"RED   : PD >= {t_red:.4f}")
print("==========================\n")

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
df_raw['y'] = df_raw.get('default_status', 0)

# Pull 5 defaulting and 5 non-defaulting
bad = df_raw[df_raw['y'] == 1].sample(5, random_state=42)
good = df_raw[df_raw['y'] == 0].sample(5, random_state=42)
test_set = pd.concat([bad, good]).sample(frac=1, random_state=1)
X_test = X.loc[test_set.index]

xgb_feats = m_xgb.get_booster().feature_names
X_test = X_test[xgb_feats]

p_xgb = m_xgb.predict_proba(X_test)[:, 1]
p_lgbm = m_lgbm.predict_proba(X_test)[:, 1]
X_m = m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm}))
p_final = m_meta.predict_proba(X_m)[:, 1]

print("=== 10 TEST CASES ===")
for i, (idx, row) in enumerate(test_set.iterrows()):
    prob = p_final[i]
    if prob >= t_red: band = 'RED   '
    elif prob >= t_green: band = 'YELLOW'
    else: band = 'GREEN '
    
    act = 'Default' if row['y'] == 1 else 'Paid   '
    f1 = X_test.loc[idx, 'stress_f1']
    f5 = X_test.loc[idx, 'stress_f5'] * 5
    f6 = X_test.loc[idx, 'stress_f6'] * 10
    print(f"[{i+1:02d}] True: {act} | Env. PD: {prob:.4f} | Band: {band} | Profiles -> EMI/Inc: {f1:.2f}, Bounces: {f5:.0f}, Loan Apps: {f6:.0f}")
