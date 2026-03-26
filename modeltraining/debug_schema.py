from supabase import create_client, Client
import json

# Credentials
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def debug_columns():
    # Try inserting with different common names
    test_cases = [
        {"external_id": "DEBUG_1"},
        {"customer_id": "DEBUG_2"},
        {"external_customer_id": "DEBUG_3"}
    ]
    
    for case in test_cases:
        try:
            print(f"Testing insert with key: {list(case.keys())[0]}")
            res = supabase.table("customers").insert(case).execute()
            print(f"SUCCESS with {list(case.keys())[0]}")
            # Clean up
            supabase.table("customers").delete().eq(list(case.keys())[0], list(case.values())[0]).execute()
            break
        except Exception as e:
            print(f"FAILED with {list(case.keys())[0]}: {str(e)}")

if __name__ == "__main__":
    debug_columns()
