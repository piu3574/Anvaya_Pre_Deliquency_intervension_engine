import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def export_audit_tables():
    print("🚀 Exporting Live Audit Tables from Supabase...")
    
    # 1. Export risk_scores
    try:
        res_scores = supabase.table("risk_scores").select("*").limit(1000).execute()
        if res_scores.data:
            df_scores = pd.DataFrame(res_scores.data)
            df_scores.to_csv("risk_scores_audit_export.csv", index=False)
            print("   [+] Exported risk_scores_audit_export.csv")
        else:
            print("   [-] risk_scores table is empty.")
    except Exception as e:
        print(f"   [!] Error exporting risk_scores: {e}")

    # 2. Export risk_explanations
    try:
        res_shap = supabase.table("risk_explanations").select("*").limit(1000).execute()
        if res_shap.data:
            df_shap = pd.DataFrame(res_shap.data)
            df_shap.to_csv("risk_explanations_audit_export.csv", index=False)
            print("   [+] Exported risk_explanations_audit_export.csv")
        else:
            print("   [-] risk_explanations table is empty.")
    except Exception as e:
        print(f"   [!] Error exporting risk_explanations: {e}")

if __name__ == "__main__":
    export_audit_tables()
