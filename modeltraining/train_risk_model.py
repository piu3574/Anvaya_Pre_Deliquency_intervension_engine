import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    accuracy_score, precision_recall_curve, auc
)
from scipy.stats import ks_2samp

# 1. CONFIG & REPRODUCIBILITY
SEED = 42
np.random.seed(SEED)
DATA_PATH = "dataset/hyper_realistic_portfolio_100k.csv"
OUTPUT_DIR = os.path.join("modeltraining", "artifacts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. FEATURE ENGINEERING (TURBO SIGNATURE)
def engineer_turbo_features(df_raw):
    """
    Creates the 'Turbo' 10-feature Decisive Signal set.
    """
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
    y = df_raw.get('default_status', None)
    return X, y

# 3. MAIN WORKFLOW
def run_final_turbo_pipeline():
    print("🚀 [Step 1] Loading Hyper-Realistic Data...")
    df_raw = pd.read_csv(DATA_PATH)
    X, y = engineer_turbo_features(df_raw)
    
    print("🚀 [Step 2] 70:30 Stratified Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    
    print("🚀 [Step 3] Training Base Models (XGB & LGBM)...")
    m_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric='logloss', scale_pos_weight=5.0)
    m_xgb.fit(X_train, y_train)
    
    m_lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, scale_pos_weight=5.0)
    m_lgbm.fit(X_train, y_train)
    
    print("🚀 [Step 4] Training Meta-Model (Stacking)...")
    # Generate meta-features (PDs) from train set
    pd_xgb_tr = m_xgb.predict_proba(X_train)[:, 1]
    pd_lgbm_tr = m_lgbm.predict_proba(X_train)[:, 1]
    X_meta_tr = pd.DataFrame({'xgb': pd_xgb_tr, 'lgbm': pd_lgbm_tr})
    
    m_scaler = StandardScaler()
    X_meta_tr_s = m_scaler.fit_transform(X_meta_tr)
    
    m_meta_base = LogisticRegression(random_state=SEED)
    m_meta_base.fit(X_meta_tr_s, y_train)
    
    m_meta = CalibratedClassifierCV(estimator=m_meta_base, method='isotonic', cv=5)
    m_meta.fit(X_meta_tr_s, y_train)
    
    print("🚀 [Step 5] Final Multi-Model Evaluation...")
    p_xgb = m_xgb.predict_proba(X_test)[:, 1]
    p_lgbm = m_lgbm.predict_proba(X_test)[:, 1]
    X_meta_ts = m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm}))
    p_final = m_meta.predict_proba(X_meta_ts)[:, 1]
    
    # Threshold for Accuracy (Fixed explicitly by User for App)
    t_green = 0.1250
    t_red = 0.2500
    
    def eval_model(y_true, y_prob, name):
        y_pred = (y_prob >= t_red).astype(int)
        auc_v = roc_auc_score(y_true, y_prob)
        gini = 2 * auc_v - 1
        ks = ks_2samp(y_prob[y_true==1], y_prob[y_true==0]).statistic
        return {
            "Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred),
            "AUC": auc_v,
            "Gini": gini,
            "KS": ks
        }
    
    results = [
        eval_model(y_test, p_xgb, "XGBoost"),
        eval_model(y_test, p_lgbm, "LightGBM"),
        eval_model(y_test, p_final, "Ensemble (Stack)")
    ]
    df_res = pd.DataFrame(results)
    print("\n--- FINAL SCORECARD ---")
    print(df_res.to_markdown(index=False))
    
    print("\n--- META-MODEL COEFFICIENTS ---")
    print(f"Intercept (a): {m_meta_base.intercept_[0]:.6f}")
    print(f"Weight XGB (w1): {m_meta_base.coef_[0][0]:.6f}")
    print(f"Weight LGBM (w2): {m_meta_base.coef_[0][1]:.6f}")
    print(f"Optimized Red Threshold: {t_red:.4f}")

    # 🚀 [Step 6] Persist Everything
    joblib.dump(m_xgb, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
    joblib.dump(m_lgbm, os.path.join(OUTPUT_DIR, "lgbm_model.pkl"))
    joblib.dump(m_meta, os.path.join(OUTPUT_DIR, "ensemble_meta.pkl"))
    joblib.dump(m_scaler, os.path.join(OUTPUT_DIR, "meta_scaler.pkl"))
    
    banding_config = {'green': t_green*100, 'red': t_red*100}
    with open(os.path.join(OUTPUT_DIR, "banding_config.pkl"), "wb") as f:
        pickle.dump(banding_config, f)
    
    print("\n✅ Turbo Pipeline Finalized and Saved for Hackathon.")

if __name__ == "__main__":
    run_final_turbo_pipeline()
