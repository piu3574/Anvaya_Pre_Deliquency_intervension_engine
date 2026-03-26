from supabase import create_client, Client
import json

# Credentials
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def audit():
    print("Executing Supabase Health Audit...")
    tables = [
        "dashboard_customers",
        "customers",
        "risk_scores",
        "risk_explanations",
        "alerts",
        "loans_raw",
        "balances_raw"
    ]
    
    report = {}
    for t in tables:
        try:
            res = supabase.table(t).select("*", count="exact").limit(1).execute()
            report[t] = {"count": res.count, "status": "LIVE"}
        except Exception as e:
            report[t] = {"count": 0, "status": f"UNKNOWN/ERROR: {str(e)}"}
            
    print("\n--- Health Report ---")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    audit()
