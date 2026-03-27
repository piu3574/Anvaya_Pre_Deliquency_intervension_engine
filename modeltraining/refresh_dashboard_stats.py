import pandas as pd
from supabase import create_client, Client
import os
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Credentials
SUPARTIFACTS_DIR = os.path.join("modeltraining", "artifacts")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_dashboard_data():
    print("Loading dashboard_data.csv...")
    df = pd.read_csv("dashboard_data.csv")
    
    # Force deduplication to ensure unique Primary Keys
    val_before = len(df)
    df.drop_duplicates(subset='customer_id', keep='first', inplace=True)
    val_after = len(df)
    if val_before != val_after:
        print(f"Dropped {val_before - val_after} duplicate IDs.")

    # Supabase has a limit on the size of a single insert.
    # We will upload in batches of 500.
    batch_size = 500
    total_rows = len(df)
    
    print(f"Uploading {total_rows} rows in batches of {batch_size}...")
    
    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i+batch_size].to_dict(orient="records")
        try:
            supabase.table("dashboard_customers").upsert(batch).execute()
            print(f"   Uploaded rows {i} to {min(i+batch_size, total_rows)}")
        except Exception as e:
            print(f"!!! Error at batch {i}: {e}")
            # Simple retry logic
            time.sleep(2)
            supabase.table("dashboard_customers").upsert(batch).execute()

    print("\n--- UPLOAD COMPLETE ---")

if __name__ == "__main__":
    upload_dashboard_data()
