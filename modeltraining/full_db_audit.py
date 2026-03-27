import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def full_audit():
    print("📋 FULL SUPABASE AUDIT 📋")
    
    # List of tables we suspect exist
    tables = [
        "customers", "risk_scores", "risk_explanations", "customer_features",
        "dashboard_customers", "transactions", "accounts", "balances",
        "loans_raw", "balances_raw", "accounts_raw", "savings_raw", "salary_raw",
        "model_health", "dashboard_data"
    ]
    
    report = []
    for t in tables:
        try:
            res = supabase.table(t).select("count", count="exact").limit(1).execute()
            count = res.count if res.count is not None else 0
            report.append({"table": t, "status": "EXISTS", "rows": count})
        except Exception as e:
            report.append({"table": t, "status": "MISSING/ERROR", "rows": 0})

    with open("modeltraining/db_audit_results.txt", "w") as f:
        f.write("-" * 40 + "\n")
        f.write(f"{'Table Name':<25} | {'Status':<10} | {'Rows':<10}\n")
        f.write("-" * 40 + "\n")
        for r in report:
            f.write(f"{r['table']:<25} | {r['status']:<10} | {r['rows']:<10}\n")
        f.write("-" * 40 + "\n")
    
    print("✅ Audit complete. Results in modeltraining/db_audit_results.txt")

if __name__ == "__main__":
    full_audit()
