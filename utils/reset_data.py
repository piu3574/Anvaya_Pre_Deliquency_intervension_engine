import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def reset_tables():
    print("Resetting tables for clean upload...")
    try:
        supabase.table("customer_features").delete().neq("f1", -99999).execute()
        print("   [x] Cleared customer_features")
    except: pass
    
    try:
        supabase.table("customers").delete().neq("external_id", "TRASH_ID").execute()
        print("   [x] Cleared customers")
    except: pass

if __name__ == "__main__":
    reset_tables()
