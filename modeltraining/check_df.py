import pandas as pd
df = pd.read_csv("dataset/portfolio_100k.csv")
print(f"Total Rows: {len(df)}")
print(f"Unique IDs: {df['external_customer_id'].nunique()}")
print(f"Duplicates: {len(df) - df['external_customer_id'].nunique()}")
