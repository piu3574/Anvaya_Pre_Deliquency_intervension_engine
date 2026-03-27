import os
from typing import Any, Dict, Optional, Tuple

from supabase import Client, create_client


SUPABASE_URL = os.getenv(
    "SUPABASE_URL", "https://fotkkamptuylqubvwyom.supabase.co"
).strip()
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZvdGtrYW1wdHV5bHF1YnZ3eW9tIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NDA3Nzc0NSwiZXhwIjoyMDg5NjUzNzQ1fQ.MOcNYq6s-WyjTY1T-_4QL9rlCsSgMHmE7uJYA2KS6Ig",
).strip()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_raw_features(
    customer_id: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        res = (
            supabase.table("anvaya_test_raw")
            .select("*")
            .eq("customer_id", customer_id)
            .limit(1)
            .execute()
        )
        if res.data:
            return res.data[0], None
        return None, None
    except Exception as exc:
        return None, f"Failed to fetch raw features: {exc}"
