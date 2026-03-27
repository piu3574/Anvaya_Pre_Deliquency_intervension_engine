import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def init_schema():
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # We use RPC or just raw SQL via a trick if possible, 
    # but usually we just assume the table exists or we provide the SQL.
    # I will provide the SQL for the user to run in the dashboard, 
    # and also attempt to check if it exists.
    
    print("--- SQL FOR SUPABASE DASHBOARD ---")
    print("""
CREATE TABLE IF NOT EXISTS customer_features (
    customer_id TEXT PRIMARY KEY,
    f1 NUMERIC, f2 NUMERIC, f3 NUMERIC, f4 NUMERIC, f5 NUMERIC, f6 NUMERIC, 
    f7 NUMERIC, f8 NUMERIC, f9 NUMERIC, f10 NUMERIC, f12 NUMERIC, f13 NUMERIC, f14 NUMERIC,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
    """)
    print("----------------------------------")

if __name__ == "__main__":
    init_schema()
