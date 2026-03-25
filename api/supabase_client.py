import os
from typing import Dict, Any, List
from supabase import create_client, Client
import datetime

# Credentials read from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://your-project-id.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your-anon-or-service-role-key")

# Initialize client lazily or defensively
supabase = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL.startswith("http"):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_raw_features(customer_id: str) -> Dict[str, Any]:
    """
    Fetches raw F1-F13 features for a specific customer from Supabase.
    """
    if not supabase:
        print("Supabase client not initialized. Check environment variables.")
        return None
    try:
        response = supabase.table("customer_features")\
            .select("F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F12, F13, F14")\
            .eq("customer_id", customer_id)\
            .limit(1)\
            .execute()
        
        if not response.data:
            return None
        return response.data[0]
    except Exception as e:
        print(f"Error fetching features for {customer_id}: {e}")
        return None

def log_score(customer_id: str, pd: float, band: str, reason_codes: List[str]):
    """
    Logs the PD score, risk band, and reason codes to the model_scores table.
    """
    try:
        data = {
            "customer_id": customer_id,
            "pd_value": pd,
            "risk_band": band,
            "top_reason_codes": reason_codes,
            "scored_at": datetime.datetime.utcnow().isoformat()
        }
        supabase.table("model_scores").upsert(data).execute()
    except Exception as e:
        print(f"Error logging score for {customer_id}: {e}")
