import os
from typing import Dict, Any, List, Optional
from supabase import create_client, Client
import datetime

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://your-project-id.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your-anon-or-service-role-key")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL.startswith("http"):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_raw_features(customer_id: str) -> Optional[Dict[str, Any]]:
    """Fetches raw F1-F14 features for a specific customer from Supabase."""
    if not supabase:
        print("Supabase client not initialized. Check environment variables.")
        return None
    try:
        response = (
            supabase.table("customer_features")
            .select("*")
            .eq("customer_id", customer_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            return None
        return response.data[0]
    except Exception as e:
        print(f"Error fetching features for {customer_id}: {e}")
        return None

def get_all_customers(limit: int = 200) -> Optional[List[Dict[str, Any]]]:
    """Fetches all customers (with features) from Supabase customer_features table."""
    if not supabase:
        print("Supabase client not initialized. Check environment variables.")
        return []
    try:
        response = (
            supabase.table("customer_features")
            .select("*")
            .limit(limit)
            .execute()
        )
        return response.data or []
    except Exception as e:
        print(f"Error fetching all customers: {e}")
        return []

def log_score(customer_id: str, pd: float, band: str, reason_codes: List[str]):
    """Logs the PD score, risk band, and reason codes to the model_scores table."""
    if not supabase:
        return
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
