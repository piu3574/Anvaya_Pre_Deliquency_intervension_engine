import pandas as pd
import numpy as np
import json

def optimize():
    print("Analyzing PD vs Actual Default Rate (100k Portfolio)...")
    df = pd.read_csv("dashboard_data.csv")
    
    # Create 5% width buckets for PD
    df['pd_bucket'] = (df['pd_final'] // 0.05) * 0.05
    
    analysis = df.groupby('pd_bucket').agg(
        count=('y_true', 'count'),
        actual_dr=('y_true', 'mean'),
        avg_pd=('pd_final', 'mean')
    ).reset_index()
    
    print("\n--- Risk Transition Analysis ---")
    print(analysis.to_string(index=False))
    
    # Heuristic for neat thresholds based on DR spikes
    # Let's say:
    # GREEN: DR remains < 2%
    # YELLOW: DR remains < 10%
    # RED: DR > 10% spike
    
    thresholds = {
        "green_limit": 0.0,
        "yellow_limit": 0.0
    }
    
    for i, row in analysis.iterrows():
        if row['actual_dr'] > 0.02 and thresholds["green_limit"] == 0.0:
            thresholds["green_limit"] = round(row['pd_bucket'], 2)
        if row['actual_dr'] > 0.10 and thresholds["yellow_limit"] == 0.0:
            thresholds["yellow_limit"] = round(row['pd_bucket'], 2)
            
    # If not found, use defaults
    if thresholds["green_limit"] == 0.0: thresholds["green_limit"] = 0.10
    if thresholds["yellow_limit"] == 0.0: thresholds["yellow_limit"] = 0.25
    
    print(f"\nRecommended Data-Driven Thresholds:")
    print(f"GREEN  : PD < {thresholds['green_limit']:.0%}")
    print(f"YELLOW : {thresholds['green_limit']:.0%} <= PD < {thresholds['yellow_limit']:.0%}")
    print(f"RED    : PD >= {thresholds['yellow_limit']:.0%}")
    
    with open("modeltraining/optimized_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

if __name__ == "__main__":
    optimize()
