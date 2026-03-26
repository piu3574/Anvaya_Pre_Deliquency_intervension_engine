import pandas as pd
import numpy as np
import json

def diagnose():
    print("Loading dashboard_data.csv for diagnostic...")
    df = pd.read_csv("dashboard_data.csv")
    
    # 1. Score Distribution
    scores = df['pd_final']
    dist = {
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "median": float(scores.median()),
        "p25": float(scores.quantile(0.25)),
        "p75": float(scores.quantile(0.75)),
        "p90": float(scores.quantile(0.90)),
        "p95": float(scores.quantile(0.95))
    }
    
    # 2. Band Shares
    shares = df['risk_band'].value_counts(normalize=True).to_dict()
    
    # 3. Calibration by Band
    calib = {}
    for band in ['GREEN', 'YELLOW', 'RED']:
        subset = df[df['risk_band'] == band]
        if len(subset) > 0:
            calib[band] = {
                "count": int(len(subset)),
                "default_rate": float(subset['y_true'].mean()),
                "avg_pd": float(subset['pd_final'].mean())
            }
        else:
            calib[band] = {"count": 0, "default_rate": 0.0, "avg_pd": 0.0}

    # 4. Feature Analysis for Bias
    # Let's check a few critical risk indicators in the test set
    feature_stats = {
        "avg_f1_emi_ratio": float(df['f1_emi_to_income'].mean()),
        "median_f5_failed_debits": float(df['f5_auto_debit_fails'].median()),
        "pct_with_failed_debits": float((df['f5_auto_debit_fails'] > 0).mean()),
        "total_default_rate": float(df['y_true'].mean())
    }

    results = {
        "score_distribution": dist,
        "band_shares": shares,
        "calibration_by_band": calib,
        "feature_stats": feature_stats
    }
    
    print(json.dumps(results, indent=2))
    with open("modeltraining/diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    diagnose()
