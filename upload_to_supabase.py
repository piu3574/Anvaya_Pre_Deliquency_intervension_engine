import os
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
from supabase import create_client, Client
from sklearn.model_selection import train_test_split

os.chdir(r"C:\Users\saksh\Documents\Anvaya")

SUPABASE_URL = "https://ckcxagnpxypowuswptir.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNrY3hhZ25weHlwb3d1c3dwdGlyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDU0MDE2NywiZXhwIjoyMDkwMTE2MTY3fQ.IZqc1luExz4iLTCLKxQJxiNF1yQmx-f4WoHH6pJsTRk"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# LOAD
m_xgb = joblib.load('modeltraining/artifacts/xgb_model.pkl')
m_lgbm = joblib.load('modeltraining/artifacts/lgbm_model.pkl')
m_meta = joblib.load('modeltraining/artifacts/ensemble_meta.pkl')
m_scaler = joblib.load('modeltraining/artifacts/meta_scaler.pkl')

bc = pickle.load(open('modeltraining/artifacts/banding_config.pkl', 'rb'))
t_green = bc['green']/100.0
t_red = bc['red']/100.0

df_raw = pd.read_csv('dataset/hyper_realistic_portfolio_100k.csv')

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
X.fillna(0, inplace=True)

# 30k test split EXACTLY
df_train, df_test_raw, _, y_test = train_test_split(df_raw, df_raw['y'], test_size=0.3, stratify=df_raw['y'], random_state=42)
X_test = X.loc[df_test_raw.index][m_xgb.get_booster().feature_names]

def S(col, default=0):
    val = df_test_raw.get(col, default)
    if isinstance(val, (int, float, str, bool)):
        return pd.Series([val]*len(df_test_raw), index=df_test_raw.index)
    return val

# RAW TABLE
raw_upload = pd.DataFrame({
    'customer_id': df_test_raw['customer_id'],
    'age': S('age').fillna(0).astype(int),
    'gender': S('gender', 'UNKNOWN'),
    'customer_type': S('employment_category', 'UNKNOWN'),
    'employment_sector': S('employment_sector', 'UNKNOWN'),
    'region': S('region', 'UNKNOWN'),
    
    'primary_income_monthly': S('monthly_net_salary').fillna(0).round(2),
    'has_secondary_income': (S('total_monthly_income') > S('monthly_net_salary')).astype(bool),
    'secondary_income_typical_monthly': (S('total_monthly_income') - S('monthly_net_salary')).fillna(0).round(2),
    
    'has_credit_card': S('has_credit_card', False).astype(bool),
    'credit_card_limit_total': S('credit_card_limit_total').fillna(0).round(2),
    'has_overdraft_facility': S('has_overdraft_facility', False).astype(bool),
    'overdraft_limit': S('overdraft_limit').fillna(0).round(2),
    
    'num_active_loans': S('number_of_active_loans').fillna(0).astype(int),
    'total_loan_principal': S('total_loan_outstanding').fillna(0).round(2),
    'total_monthly_emi_due': S('total_monthly_emi_amount').fillna(0).round(2),
    
    'current_balance': S('savings_balance_current').fillna(0).round(2),
    'balance_60d_ago': S('savings_balance_60d_ago').fillna(0).round(2),
    'expected_salary_day': S('expected_salary_date', 1).fillna(1).astype(int),
    
    'num_auto_debit_failures_last_30d': S('auto_debit_failure_count_30d').fillna(0).astype(int),
    'num_lending_app_txn_last_30d': S('lending_app_transaction_count_30d').fillna(0).astype(int),
    'num_overdraft_days_last_30d': S('overdraft_days_30d').fillna(0).astype(int),
    
    'default_label': y_test.astype(int)
})

p_xgb = m_xgb.predict_proba(X_test)[:,1]
p_lgbm = m_lgbm.predict_proba(X_test)[:,1]
p_final = m_meta.predict_proba(m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm})))[:,1]

y_pred = (p_final >= t_red).astype(int)

def get_confusion(y_t, y_p):
    if y_t == 1 and y_p == 1: return 'TP'
    if y_t == 0 and y_p == 1: return 'FP'
    if y_t == 0 and y_p == 0: return 'TN'
    return 'FN'
    
bands = np.where(p_final >= t_red, 'RED', np.where(p_final >= t_green, 'YELLOW', 'GREEN'))
conf_buckets = [get_confusion(yt, yp) for yt, yp in zip(y_test, y_pred)]

# SHAP values
print("Computing SHAP values for 30k test set...")
explainer = shap.TreeExplainer(m_lgbm)
shap_v = explainer.shap_values(X_test)
features = X_test.columns.tolist()

res_upload = []
for i in range(len(X_test)):
    cust = df_test_raw.iloc[i]['customer_id']
    sv = shap_v[i]
    
    top_indices = np.argsort(np.abs(sv))[::-1][:5]
    shap_json = {features[idx]: round(float(sv[idx]), 4) for idx in top_indices}
    
    pos_idx = np.argmax(sv)
    neg_idx = np.argmin(sv)
    
    res_upload.append({
        'customer_id': cust,
        'model_version': '3.0.0-turbo',
        'xgb_model_hash': 'v3-xgb-01',
        'lgbm_model_hash': 'v3-lgb-01',
        'ensemble_model_hash': 'v3-meta-01',
        'pd_xgboost': round(float(p_xgb[i]), 4),
        'pd_lightgbm': round(float(p_lgbm[i]), 4),
        'pd_final': round(float(p_final[i]), 4),
        'risk_band': bands[i],
        'predicted_default_flag': bool(y_pred[i]),
        'true_default_label': int(y_test.iloc[i]),
        'confusion_bucket': conf_buckets[i],
        'shap_top_features': shap_json,
        'top_positive_driver': features[pos_idx] if sv[pos_idx] > 0 else None,
        'top_negative_driver': features[neg_idx] if sv[neg_idx] < 0 else None
    })

res_df = pd.DataFrame(res_upload)

raw_upload = raw_upload.replace({np.nan: None})
raw_upload = raw_upload.replace([np.inf, -np.inf], None)

res_df = res_df.replace({np.nan: None})
res_df = res_df.replace([np.inf, -np.inf], None)

def push_table(df, name):
    records = df.to_dict(orient='records')
    print(f"\nStarting {name} upload ({len(records)} rows)...")
    for i in range(0, len(records), 500):
        batch = records[i:i+500]
        try:
            supabase.table(name).upsert(batch).execute()
            print(f"[{name}] Inserted {i+len(batch)} / {len(records)}")
        except Exception as e:
            print(f"[{name}] ERROR on batch {i}: {e}")

push_table(raw_upload, 'anvaya_test_raw')
push_table(res_df, 'anvaya_test_results')
print("\n✅ SUPABASE SYNCHRONIZATION COMPLETE.")
