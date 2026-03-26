from supabase import create_client, Client
import os

# Credentials
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def clear_table():
    print("Clearing dashboard_customers table...")
    # Using a filter that matches all rows (e.g., customer_id not equal to something impossible)
    res = supabase.table("dashboard_customers").delete().neq("customer_id", "000000000").execute()
    print("Table cleared.")

if __name__ == "__main__":
    clear_table()
