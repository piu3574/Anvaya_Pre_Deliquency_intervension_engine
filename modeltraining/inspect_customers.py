from supabase import create_client, Client
import json

# Credentials
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def inspect_customers():
    print("Inspecting 'customers' table contents...")
    res = supabase.table("customers").select("*").limit(5).execute()
    print(json.dumps(res.data, indent=2))

if __name__ == "__main__":
    inspect_customers()
