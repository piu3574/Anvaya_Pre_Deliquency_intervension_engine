import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("Checking API Health...")
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    print(f"Health: {res.json()}")

def test_high_risk_scenario():
    """
    Simulates a high-risk customer (TEST_RED_1)
    Logic: High EMI/Income, multiple failed debits, low buffer.
    """
    customer_id = "TEST123" # Reusing existing test ID with known high stress in DB (assumed)
    print(f"\nEvaluating High Risk Scenario for {customer_id}...")
    
    start_time = time.time()
    res = requests.get(f"{BASE_URL}/score/{customer_id}")
    duration = time.time() - start_time
    
    if res.status_code == 200:
        data = res.json()
        print(f"Result: PD_Final={data['pd_final']}, Band={data['band']}")
        print(f"Top Drivers: {data['top_drivers']}")
        print(f"Latency: {duration:.2f}s")
        
        # Validation
        if data['pd_final'] > 0.05: # Threshold check (example)
            print("✅ Logic Check: High PD detected.")
        else:
            print("⚠️ Logic Check: PD seems lower than expected for stress profile.")
    else:
        print(f"❌ Error {res.status_code}: {res.text}")

def test_batch_processing():
    print("\nTesting Batch Scoring...")
    payload = {"customer_ids": ["TEST123", "TEST234", "TEST345"]}
    res = requests.post(f"{BASE_URL}/score/batch", json=payload)
    if res.status_code == 200:
        print(f"Batch Results Count: {len(res.json())}")
        print("✅ Batch OK")
    else:
        print(f"❌ Batch Error: {res.text}")

if __name__ == "__main__":
    try:
        test_health()
        test_high_risk_scenario()
        test_batch_processing()
        print("\n--- All Scenario Tests Triggered ---")
    except Exception as e:
        print(f"Scenario Test Failed: {e}")
