"""
compute_thresholds.py
Computes data-driven PD thresholds for 3-band risk banding.
- Reads the scored dataset.
- Buckets the PD_final into deciles.
- Shows actual default rate per decile.
- Recommends MEDIUM/HIGH split threshold where actual default rate
  crosses the ~15% line (matching the reference image logic).
"""
import pandas as pd
import numpy as np

df = pd.read_csv("modeltraining/anvaya_scored_dataset.csv")

# Only look at non-GREEN rows (those that went through LightGBM)
df_deep = df[df['triage'] != 'GREEN'].copy()

print(f"Total rows          : {len(df):,}")
print(f"GREEN (XGB exit)    : {(df['triage']=='GREEN').sum():,}  ({(df['triage']=='GREEN').mean()*100:.1f}%)")
print(f"Non-GREEN (deep)    : {len(df_deep):,}  ({len(df_deep)/len(df)*100:.1f}%)")
print(f"\nDefault rate GREEN  : {df[df['triage']=='GREEN']['default_flag'].mean()*100:.2f}%")
print(f"Default rate non-GREEN: {df_deep['default_flag'].mean()*100:.2f}%")

print("\n--- PD Bucket Analysis (non-GREEN rows only) ---")
print(f"{'PD Range':15s} {'Customers':>10s} {'Defaulted':>10s} {'Actual DR%':>12s}")
print("-" * 52)

bins  = [0, 5, 10, 15, 20, 25, 30, 40, 50, 100]
labels = ["0-5%","5-10%","10-15%","15-20%","20-25%","25-30%","30-40%","40-50%","50%+"]
df_deep['pd_bucket'] = pd.cut(df_deep['PD_final'], bins=bins, labels=labels)

for lbl in labels:
    sub = df_deep[df_deep['pd_bucket'] == lbl]
    if len(sub) == 0: continue
    dr = sub['default_flag'].mean() * 100
    print(f"{lbl:15s} {len(sub):>10,} {sub['default_flag'].sum():>10,} {dr:>11.1f}%")

# Find natural breakpoint: where actual default rate crosses 15%
print("\n--- Recommended MEDIUM / HIGH threshold ---")
for thresh in [10, 12, 15, 18, 20]:
    med   = df_deep[df_deep['PD_final'] < thresh]
    high  = df_deep[df_deep['PD_final'] >= thresh]
    if len(med) == 0 or len(high) == 0: continue
    print(f"Threshold {thresh:2d}%:  MEDIUM={len(med):,} (DR={med['default_flag'].mean()*100:.1f}%)  "
          f"HIGH={len(high):,} (DR={high['default_flag'].mean()*100:.1f}%)")
