import os
from typing import Dict, Any, List, Optional, Tuple
from supabase import create_client, Client
import datetime

# Credentials (Hardcoded as requested)
SUPABASE_URL = "https://fotkkamptuylqubvwyom.supabase.co".strip()
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig".strip()

# Initialize client lazily or defensively
supabase = None
if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL.startswith("http"):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_raw_features(customer_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Fetches raw F1-F13 features for a specific customer from Supabase.
    Returns (data, error_message).
    """
    if not supabase:
        return None, "Supabase client not initialized."
    try:
        response = supabase.table("customer_features")\
            .select("f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f12, f13, f14")\
            .eq("customer_id", customer_id)\
            .limit(1)\
            .execute()
        
        if not response.data:
            return None, None # Truly not found
        return response.data[0], None
    except Exception as e:
        err_msg = str(e)
        print(f"Error fetching features for {customer_id}: {err_msg}")
        return None, err_msg

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
        supabase.table("risk_scores").upsert(data).execute()
    except Exception as e:
        print(f"Error logging score for {customer_id}: {e}")
