import pandas as pd
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import time

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def populate_master_customers():
    print("1. Loading 100k Portfolio for ID extraction...")
    df = pd.read_csv("dataset/portfolio_100k.csv")
    unique_ids = df['external_customer_id'].unique().tolist()
    
    print(f"2. Preparing Registry payload for {len(unique_ids)} customers...")
    payload = [{"external_id": cid} for cid in unique_ids]

    BATCH_SIZE = 2000
    print(f"3. Uploading Registry in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(payload), BATCH_SIZE):
        batch = payload[i : i + BATCH_SIZE]
        try:
            # Upsert on external_id
            supabase.table("customers").upsert(batch, on_conflict="external_id").execute()
            print(f"   [+] Registry: {i + len(batch)} / {len(payload)}")
        except Exception as e:
            print(f"   [!] Registry Error at batch {i}: {e}")
            time.sleep(1)

    print("Master Registry (customers) populated.")

if __name__ == "__main__":
    populate_master_customers()
