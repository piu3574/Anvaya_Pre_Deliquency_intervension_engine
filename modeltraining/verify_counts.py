import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

s = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def verify():
    # Attempt a raw count and a selective count
    print("Checking counts...")
    
    # 1. Customers
    try:
        res_c = s.table("customers").select("id", count="exact").limit(1).execute()
        print(f"Customers Exact Count: {res_c.count}")
    except:
        print("Customers Exact Count Failed.")

    # 2. Features
    try:
        res_f = s.table("customer_features").select("customer_id", count="exact").limit(1).execute()
        print(f"Features Exact Count: {res_f.count}")
    except:
        print("Features Exact Count Failed.")

    # 3. Check first 5 features
    try:
        data_f = s.table("customer_features").select("customer_id").limit(5).execute()
        print(f"Features Data Sample: {len(data_f.data)} rows fetched")
        if data_f.data:
            print(f"Sample IDs: {[d['customer_id'] for d in data_f.data]}")
    except:
        print("Features Data Fetch Failed.")

if __name__ == "__main__":
    verify()
