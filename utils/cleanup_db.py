import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def cleanup_legacy_tables():
    tables_to_drop = [
        "balances", "balances_raw", "loans_raw", 
        "accounts_raw", "savings_raw", "salary_raw", 
        "model_health", "dashboard_data"
    ]
    
    print(f"🧹 Starting cleanup of {len(tables_to_drop)} legacy tables...")
    
    # Supabase Python client doesn't support DROP TABLE directly.
    # I will provide the SQL and attempt to clear the data if possible, 
    # but the user should run the DROP SQL in the dashboard for a full purge.
    
    print("--- SQL TO RUN IN SUPABASE SQL EDITOR ---")
    for table in tables_to_drop:
        print(f"DROP TABLE IF EXISTS {table} CASCADE;")
    print("-----------------------------------------")
    
    # I will attempt to delete rows from these tables to "Empty" them at least.
    for table in tables_to_drop:
        try:
            # delete() without filters fails in some Supabase setups, but we try ne 'id' 0
            supabase.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            print(f"   [x] Cleared data from {table}")
        except Exception as e:
            print(f"   [!] Could not empty {table} via API (Needs manual DROP): {e}")

if __name__ == "__main__":
    cleanup_legacy_tables()
