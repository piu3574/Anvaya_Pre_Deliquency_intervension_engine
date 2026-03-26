import pandas as pd
from supabase import create_client, Client
import os

# Credentials
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def seed_operational_customers():
    print("Loading 1,000 operational customers from portfolio...")
    df = pd.read_csv("dataset/portfolio_100k.csv").head(1000)
    
    # Map to schema: customer_id is external_customer_id
    # We rename columns to match the customers table schema if needed
    # The current schema for 'customers' usually contains raw feature data
    
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "external_id": r['external_customer_id']
        })
    
    print(f"Uploading {len(rows)} rows to 'customers' table...")
    supabase.table("customers").upsert(rows).execute()
    print("Operational seeding complete.")

if __name__ == "__main__":
    seed_operational_customers()
