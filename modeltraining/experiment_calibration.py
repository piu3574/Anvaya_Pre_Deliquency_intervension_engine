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
    accuracy_score
)

# 1. CONFIG
SEED = 42
DATA_PATH = "dataset/hyper_realistic_portfolio_100k.csv"
OUTPUT_DIR = "modeltraining"

def run_calibrated_turbo_pipeline():
    print("🚀 Loading Data...")
    df_raw = pd.read_csv(DATA_PATH)
    from train_v3_turbo_final import engineer_turbo_features
    X, y = engineer_turbo_features(df_raw)
    
    # 70:30 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    
    # Split train further for calibration vs meta-training
    X_base, X_meta_tr, y_base, y_meta_tr = train_test_split(X_train, y_train, test_size=0.4, stratify=y_train, random_state=SEED)
    
    print("🚀 Training Calibrated Base Models...")
    # XGBoost
    m_xgb_raw = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric='logloss', scale_pos_weight=5.0)
    m_xgb_cal = CalibratedClassifierCV(m_xgb_raw, method='isotonic', cv=3)
    m_xgb_cal.fit(X_base, y_base)
    
    # LightGBM
    m_lgbm_raw = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, scale_pos_weight=5.0)
    m_lgbm_cal = CalibratedClassifierCV(m_lgbm_raw, method='isotonic', cv=3)
    m_lgbm_cal.fit(X_base, y_base)
    
    print("🚀 Training Meta-Model on Calibrated PDs...")
    pd_xgb_m = m_xgb_cal.predict_proba(X_meta_tr)[:, 1]
    pd_lgbm_m = m_lgbm_cal.predict_proba(X_meta_tr)[:, 1]
    X_meta = pd.DataFrame({'xgb': pd_xgb_m, 'lgbm': pd_lgbm_m})
    
    m_scaler = StandardScaler()
    X_meta_s = m_scaler.fit_transform(X_meta)
    
    m_meta = LogisticRegression(random_state=SEED)
    m_meta.fit(X_meta_s, y_meta_tr)
    
    print("🚀 Evaluating at 30% Threshold...")
    p_xgb = m_xgb_cal.predict_proba(X_test)[:, 1]
    p_lgbm = m_lgbm_cal.predict_proba(X_test)[:, 1]
    X_ts_s = m_scaler.transform(pd.DataFrame({'xgb': p_xgb, 'lgbm': p_lgbm}))
    p_final = m_meta.predict_proba(X_ts_s)[:, 1]
    
    def report(t):
        pred = (p_final >= t).astype(int)
        print(f"--- Metrics at {t*100}% Threshold ---")
        print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
        print(f"Precision: {precision_score(y_test, pred):.4f}")
        print(f"Recall   : {recall_score(y_test, pred):.4f}")
        print(f"AUC      : {roc_auc_score(y_test, p_final):.4f}")

    report(0.30)
    
    # Optional: Save if requested
    print("\n✅ Calibration Experiment Complete.")

if __name__ == "__main__":
    run_calibrated_turbo_pipeline()
