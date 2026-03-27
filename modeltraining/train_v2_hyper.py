import pandas as pd
import numpy as np
import joblib
import pickle
import os
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    accuracy_score, precision_recall_curve, auc, brier_score_loss
)
from scipy.stats import ks_2samp, chi2
import warnings

warnings.filterwarnings('ignore')

# 1. CONFIG
DATA_PATH = "dataset/hyper_realistic_portfolio_100k.csv"
OUTPUT_DIR = "modeltraining"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEAT_COLS = [
    'F1_emi_to_income', 'F2_savings_drawdown', 'F3_salary_delay',
    'F4_spend_shift', 'F5_auto_debit_fails', 'F6_lending_app_usage',
    'F7_overdraft_freq', 'F8_stress_velocity', 'F9_payment_entropy',
    'F10_peer_stress', 'F12_cross_loan', 'F13_secondary_income',
    'F14_active_loan_pressure'
]

# 2. FEATURE ENGINEERING (REPLICATED LOGIC)
def engineer_features(df_raw):
    df = pd.DataFrame(index=df_raw.index)
    y = df_raw['default_status'] # Target updated name
    
    # F1: EMI to Income
    df['F1_emi_to_income'] = (df_raw['total_monthly_emi_amount'] / (df_raw['total_salary_credit_30d'] + 1)).clip(0, 1.5)
    
    # F2: Savings Drawdown
    df['F2_savings_drawdown'] = (df_raw['savings_balance_60d_ago'] - df_raw['savings_balance_current']) / (df_raw['savings_balance_60d_ago'] + 1)
    
    # F3: Salary Delay
    df['F3_salary_delay'] = (df_raw['salary_credit_date_m1'] - df_raw['expected_salary_date']).abs()
    
    # F4: Spend Shift
    df['F4_spend_shift'] = (df_raw['total_debit_amount_30d'] / (df_raw['total_salary_credit_30d'] + 1)).clip(0, 10)
    
    # F5: Auto Debit Fails
    df['F5_auto_debit_fails'] = df_raw['auto_debit_failure_count_30d']
    
    # F6: Lending App Usage
    df['F6_lending_app_usage'] = df_raw['lending_app_transaction_count_30d']
    
    # F7: Overdraft Freq
    df['F7_overdraft_freq'] = df_raw['overdraft_days_30d']
    
    # F8: Stress Velocity (Month end balance trend)
    df['F8_stress_velocity'] = (df_raw['savings_balance_60d_ago'] - df_raw['savings_balance_current']) / (df_raw['savings_balance_60d_ago'] + 1)
    
    # F9: Payment Entropy (Using emi_payment_day_m1 variation proxy)
    # Since we only have m1 in this dataset, we use income_volatility_ratio_3m as a proxy or stick to the column
    df['F9_payment_entropy'] = df_raw['income_volatility_ratio_3m'] * 10 
    
    # F14: Active Loan Pressure
    df['F14_active_loan_pressure'] = (df_raw['total_loan_outstanding'] / (df_raw['total_credit_limit'] + 1)).clip(0, 20)
    
    # F10: Peer Stress
    df['F10_peer_stress'] = df_raw.groupby(['employment_category'])['total_loan_outstanding'].transform('mean') / (df_raw['total_credit_limit'].mean() + 1)
    
    # F12: Cross Loan
    df['F12_cross_loan'] = df_raw['number_of_active_loans'] / (df_raw['customer_vintage_months'] + 1)
    
    # F13: Secondary Income (Inflow minus salary)
    df['F13_secondary_income'] = (df_raw['non_salary_credits_30d'] / (df_raw['total_credits_30d'] + 1)).clip(0, 1)

    return df, y

# 3. WoE CALCULATOR (REPLICATED)
def compute_woe_lookup(X_train, y_train):
    woe_lookup = {}
    manual_bins = {
        'F1_emi_to_income': [0, 0.2, 0.4, 0.6, 0.8, 1.5],
        'F5_auto_debit_fails': [0, 1, 2, 3, 5, 20],
        'F14_active_loan_pressure': [0, 0.2, 0.4, 0.6, 0.8, 2.0]
    }

    for f in FEAT_COLS:
        col_data = X_train[f]
        if f in manual_bins:
            edges = manual_bins[f]
        else:
            edges = [col_data.min()] + list(np.percentile(col_data, [20, 40, 60, 80])) + [col_data.max()]
            edges = sorted(list(set(edges)))
        
        bins = np.digitize(col_data, edges) - 1
        bins = np.clip(bins, 0, len(edges)-2)
        bins_series = pd.Series(bins, index=col_data.index)
        
        f_stats = pd.DataFrame({'bin': bins_series, 'target': y_train}).groupby('bin')['target'].agg(['count', 'sum', 'mean'])
        f_stats['non_target'] = f_stats['count'] - f_stats['sum']
        total_target = f_stats['sum'].sum()
        total_non_target = f_stats['non_target'].sum()

        f_lookup = []
        for b_idx in range(len(edges)-1):
            s = f_stats.loc[b_idx] if b_idx in f_stats.index else {'sum':0, 'non_target':0}
            t_rate = (s['sum'] + 0.5) / (total_target + 1)
            nt_rate = (s['non_target'] + 0.5) / (total_non_target + 1)
            woe_val = np.log(t_rate / nt_rate)
            f_lookup.append({
                "bin": (float(edges[b_idx]), float(edges[b_idx+1])),
                "woe": float(woe_val)
            })
        woe_lookup[f] = f_lookup
    return woe_lookup

def apply_woe(df_in, woe_lookup):
    df_out = pd.DataFrame(index=df_in.index)
    for col in FEAT_COLS:
        lookup = woe_lookup[col]
        def map_val(v):
            for entry in lookup:
                if entry['bin'][0] <= v <= entry["bin"][1]: return entry['woe']
            if v < lookup[0]['bin'][0]: return lookup[0]['woe']
            return lookup[-1]['woe']
        df_out[f"{col}_WoE"] = df_in[col].apply(map_val)
    return df_out

# 4. CALIBRATION (Hosmer-Lemeshow)
def hosmer_lemeshow_test(y_true, y_prob, bins=10):
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df['q'] = pd.qcut(df['p'], bins, duplicates='drop')
    hl_df = df.groupby('q')['y'].agg(['count', 'sum'])
    hl_df['p_avg'] = df.groupby('q')['p'].mean()
    hl_df['expected'] = hl_df['count'] * hl_df['p_avg']
    hl_df['hl_stat'] = (hl_df['sum'] - hl_df['expected'])**2 / (hl_df['expected'] * (1 - hl_df['p_avg']))
    stat = hl_df['hl_stat'].sum()
    p_val = 1 - chi2.cdf(stat, bins - 2)
    return stat, p_val, hl_df

# 5. MAIN EXECUTION
def main():
    print("1. Loading and Shuffling Data...")
    df_raw = pd.read_csv(DATA_PATH).sample(frac=1, random_state=42).reset_index(drop=True)
    X, y = engineer_features(df_raw)
    
    print("2. 70:30 Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    print("3. Fitting WoE Transformations...")
    woe_lookup = compute_woe_lookup(X_train, y_train)
    X_train_woe = apply_woe(X_train, woe_lookup)
    X_test_woe = apply_woe(X_test, woe_lookup)
    
    print("4. Training Base Models (XGB & LGBM)...")
    m_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss', scale_pos_weight=5.0)
    m_xgb.fit(X_train_woe, y_train)
    
    m_lgbm = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, scale_pos_weight=5.0)
    m_lgbm.fit(X_train_woe, y_train)
    
    print("5. Ensemble Scoring (0.4/0.6 Weighted Average)...")
    pd_xgb = m_xgb.predict_proba(X_test_woe)[:, 1]
    pd_lgbm = m_lgbm.predict_proba(X_test_woe)[:, 1]
    pd_final = (0.4 * pd_xgb) + (0.6 * pd_lgbm)
    
    # 6. THRESHOLD INFERENCE (60/30/10)
    t_green = np.percentile(pd_final, 60)
    t_red = np.percentile(pd_final, 90)
    print(f"📊 INFERRED THRESHOLDS: Green/Yellow = {t_green:.4f}, Yellow/Red = {t_red:.4f}")
    
    # 7. EVALUATION
    def get_metrics(y_true, y_prob):
        y_pred = (y_prob > t_red).astype(int)
        auc_roc = roc_auc_score(y_true, y_prob)
        gini = 2 * auc_roc - 1
        # KS
        ds = pd.DataFrame({'p': y_prob, 'y': y_true})
        ks = ks_2samp(ds[ds['y']==1]['p'], ds[ds['y']==0]['p']).statistic
        
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        
        p, r, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(r, p)
        
        return {
            "AUC-ROC": auc_roc, "Gini": gini, "KS": ks,
            "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc, "PR-AUC": pr_auc
        }

    m_xgb_met = get_metrics(y_test, pd_xgb)
    m_lgbm_met = get_metrics(y_test, pd_lgbm)
    m_final_met = get_metrics(y_test, pd_final)
    
    # 8. BUSINESS METRICS (Capture & Lift)
    def business_eval(y_true, y_prob):
        ds = pd.DataFrame({'p': y_prob, 'y': y_true}).sort_values('p', ascending=False)
        total_bad = ds['y'].sum()
        
        cap_5 = ds.head(int(len(ds)*0.05))['y'].sum() / total_bad
        cap_10 = ds.head(int(len(ds)*0.10))['y'].sum() / total_bad
        cap_20 = ds.head(int(len(ds)*0.20))['y'].sum() / total_bad
        
        # Lift at decile 1
        lift_1 = (ds.head(int(len(ds)*0.10))['y'].mean()) / (total_bad / len(ds))
        return cap_5, cap_10, cap_20, lift_1

    c5, c10, c20, l1 = business_eval(y_test, pd_final)
    
    # 9. CALIBRATION
    hl_stat, hl_p, hl_df = hosmer_lemeshow_test(y_test, pd_final)
    
    # 10. OUTPUT REPORT (Markdown Artifact)
    report = f"""# Anvaya PD Engine 2.0.0 Refinement Report
## 1. Executive Summary
- **Dataset**: `hyper_realistic_portfolio_100k.csv`
- **Total Samples**: 100,000
- **Test Default Rate**: {y_test.mean()*100:.2f}%
- **Inferred Thresholds**:
  - GREEN/YELLOW PD: **{t_green:.4f}**
  - YELLOW/RED PD  : **{t_red:.4f}**

## 2. Core Discrimination Metrics
| Metric | XGBoost | LightGBM | **Final Ensemble** |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | {m_xgb_met['AUC-ROC']:.4f} | {m_lgbm_met['AUC-ROC']:.4f} | **{m_final_met['AUC-ROC']:.4f}** |
| **Gini** | {m_xgb_met['Gini']:.4f} | {m_lgbm_met['Gini']:.4f} | **{m_final_met['Gini']:.4f}** |
| **KS Stat** | {m_xgb_met['KS']:.4f} | {m_lgbm_met['KS']:.4f} | **{m_final_met['KS']:.4f}** |
| **PR-AUC** | {m_xgb_met['PR-AUC']:.4f} | {m_lgbm_met['PR-AUC']:.4f} | **{m_final_met['PR-AUC']:.4f}** |

## 3. Class-Level Metrics (Default Class 1 @ RED Threshold)
| Metric | XGBoost | LightGBM | **Final Ensemble** |
| :--- | :--- | :--- | :--- |
| **Precision** | {m_xgb_met['Precision']:.4f} | {m_lgbm_met['Precision']:.4f} | **{m_final_met['Precision']:.4f}** |
| **Recall** | {m_xgb_met['Recall']:.4f} | {m_lgbm_met['Recall']:.4f} | **{m_final_met['Recall']:.4f}** |
| **F1-Score** | {m_xgb_met['F1']:.4f} | {m_lgbm_met['F1']:.4f} | **{m_final_met['F1']:.4f}** |
| **Accuracy** | {m_xgb_met['Accuracy']:.4f} | {m_lgbm_met['Accuracy']:.4f} | **{m_final_met['Accuracy']:.4f}** |

## 4. Calibration Analysis
**Hosmer-Lemeshow Test**: 
- Statistic: {hl_stat:.4f}
- **p-value**: {hl_p:.4e} (Lower is better calibration)

### Reliability Table
{hl_df.to_markdown()}

## 5. Business / Portfolio Impact
- **Capture Rate @ Top 5%**: {c5*100:.2f}%
- **Capture Rate @ Top 10%**: {c10*100:.2f}%
- **Capture Rate @ Top 20%**: {c20*100:.2f}%
- **Lift @ Top Decile**: {l1:.2f}x

## 6. Risk-Band Distribution (Test Set)
| Band | PD Range | Count | Distribution (%) | Default Rate |
| :--- | :--- | :--- | :--- | :--- |
| **GREEN** | < {t_green:.4f} | {len(pd_final[pd_final < t_green])} | {(len(pd_final[pd_final < t_green])/len(pd_final))*100:.1f}% | {y_test[pd_final < t_green].mean()*100:.2f}% |
| **YELLOW** | {t_green:.4f} - {t_red:.4f} | {len(pd_final[(pd_final >= t_green) & (pd_final < t_red)])} | {(len(pd_final[(pd_final >= t_green) & (pd_final < t_red)])/len(pd_final))*100:.1f}% | {y_test[(pd_final >= t_green) & (pd_final < t_red)].mean()*100:.2f}% |
| **RED** | >= {t_red:.4f} | {len(pd_final[pd_final >= t_red])} | {(len(pd_final[pd_final >= t_red])/len(pd_final))*100:.1f}% | {y_test[pd_final >= t_red].mean()*100:.2f}% |
"""
    
    with open("modeltraining/model_evaluation_report.md", "w") as f:
        f.write(report)
    
    # Persist Models
    joblib.dump(m_xgb, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
    joblib.dump(m_lgbm, os.path.join(OUTPUT_DIR, "lgbm_model.pkl"))
    with open(os.path.join(OUTPUT_DIR, "woe_lookup.pkl"), "wb") as f:
        pickle.dump(woe_lookup, f)
    
    # Save Thresholds for API
    banding_config = {'green': t_green * 100, 'red': t_red * 100}
    with open(os.path.join(OUTPUT_DIR, "banding_config.pkl"), "wb") as f:
        pickle.dump(banding_config, f)

    print("Pipeline Complete. Report saved to modeltraining/model_evaluation_report.md")

if __name__ == "__main__":
    main()
