import pandas as pd
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import time

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_100k_features():
    print("1. Loading 100k Portfolio & Deduplicating...")
    df = pd.read_csv("dataset/portfolio_100k.csv")
    df = df.drop_duplicates(subset=['external_customer_id']) # 94,605 rows
    
    print(f"2. Preparing {len(df)} unique payload...")
    payload = []
    for _, row in df.iterrows():
        item = {
            "customer_id": row['external_customer_id'],
            "f1": float(row['total_monthly_emi_amount']),
            "f2": float(row['monthly_net_salary']),
            "f3": float(row['expected_salary_day_of_month']),
            "f4": float(row['total_debit_amount_30d']),
            "f5": float(row['failed_auto_debits_m1'] + row['failed_auto_debits_m2']),
            "f6": float(row['lending_app_transaction_count_30d']),
            "f7": float(row['overdraft_days_30d']),
            "f8": float(row['end_of_month_balance_m6'] - row['end_of_month_balance_m1']),
            "f9": 0.0,
            "f10": float(row['total_loan_outstanding']),
            "f12": float(row['number_of_active_loans']),
            "f13": float(row['customer_vintage_months']),
            "f14": float(row['total_credit_limit'])
        }
        payload.append(item)

    BATCH_SIZE = 2000 
    print(f"3. Uploading {len(payload)} feature vectors...")
    
    for i in range(0, len(payload), BATCH_SIZE):
        batch = payload[i : i + BATCH_SIZE]
        try:
            # Note: Upsert on PK (customer_id) is default behavior
            supabase.table("customer_features").upsert(batch).execute()
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"   [+] Features: {i + len(batch)} / {len(payload)}")
        except Exception as e:
            print(f"   [!] Error at batch {i}: {e}")
            time.sleep(0.5)

    print("✅ 100k Feature Store migration complete.")

if __name__ == "__main__":
    upload_100k_features()
