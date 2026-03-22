import pandas as pd
df = pd.read_csv("dataset/barclays_bank_synthetic_data.csv")
print("Shape:", df.shape)
print("Nulls:", df.isna().sum().sum())
print("Default by segment:\n", df.groupby("customer_segment")["default_flag"].sum())
print("Employment:\n", df["employment_category"].value_counts(normalize=True))
print("Defaults rate total:", df["default_flag"].mean())
print("Has cold start flag:", df["cold_start_flag"].sum())
print("Max vintage:", df["customer_vintage_months"].max())
print("Checking continuous history fields exist...")
for m in [1, 6, 12]:
    assert f"monthly_net_salary_m{m}" in df.columns
    assert f"end_of_month_balance_m{m}" in df.columns
print("Verification complete.")
