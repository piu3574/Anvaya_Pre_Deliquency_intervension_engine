"""
materialize.py — Orchestrates the Feature Generation Pipeline.
Runs the Generator for all active customers and updates the 'customer_features' store.
"""
import datetime
import os
from typing import Optional, List, Dict, Any, Tuple
from api.supabase_client import supabase  # Reuse existing client
from feature_engineering.generator import FeatureGenerator

def run_materialization(snapshot_target: Optional[datetime.date] = None):
    if snapshot_target is None:
        snapshot_target = datetime.date.today()

    print(f"--- Feature Materialization Started for {snapshot_target} ---")
    gen = FeatureGenerator(supabase)

    # 1. Fetch active customers from accounts_raw
    try:
        accounts_res = supabase.table("accounts_raw").select("customer_id").execute()
        customer_ids = [r['customer_id'] for r in accounts_res.data]
    except Exception as e:
        print(f"Error fetching accounts: {e}")
        return

    print(f"Processing {len(customer_ids)} customers...")

    for cid in customer_ids:
        try:
            # 2. Compute F1-F14
            fv = gen.get_feature_vector(cid, snapshot_target)
            
            # 3. Upsert into customer_features (the store used by API)
            row = {
                "customer_id": cid,
                **fv,
                "created_at": datetime.datetime.now().isoformat()
            }
            supabase.table("customer_features").upsert(row, on_conflict="customer_id").execute()
            print(f"   [OK] {cid}")
        except Exception as e:
            print(f"   [FAIL] {cid}: {e}")

    print("--- Materialization Complete ---")

if __name__ == "__main__":
    run_materialization()
