import pandas as pd
import numpy as np
import os

def generate_100k_balanced():
    print("Generating 100,000 Balanced Portfolio Records...")
    np.random.seed(42)
    n = 100000
    
    # 1. Define Segments
    # 60% Prime, 30% Standard, 10% Risky
    segments = np.random.choice(['PRIME', 'STANDARD', 'RISKY'], n, p=[0.60, 0.30, 0.10])
    
    # Ensure unique IDs
    raw_ids = [f"CUST_{i:06d}" for i in range(1, n + 1)]
    np.random.shuffle(raw_ids)
    
    for i, seg in enumerate(segments):
        if seg == 'PRIME':
            emi_inc = np.random.uniform(0.1, 0.3)
            bounces = 0
            savings = np.random.uniform(50000, 200000)
            default_prob = 0.01 # 1% default chance
        elif seg == 'STANDARD':
            emi_inc = np.random.uniform(0.3, 0.55)
            bounces = np.random.choice([0, 1], p=[0.8, 0.2])
            savings = np.random.uniform(10000, 50000)
            default_prob = 0.08 # 8% default chance
        else: # RISKY
            emi_inc = np.random.uniform(0.55, 0.95)
            bounces = np.random.randint(1, 5)
            savings = np.random.uniform(0, 10000)
            default_prob = 0.35 # 35% default chance
            
        default_flag = 1 if np.random.random() < default_prob else 0
        
        # Derive other features with some noise
        # F1_emi_to_income is emi_inc
        # F5_auto_debit_fails is bounces
        
        row = {
            'total_monthly_emi_amount': emi_inc * 100000, # Assuming 100k income
            'monthly_net_salary': 100000,
            'savings_balance_60d_ago': savings * 1.2,
            'current_account_balance': savings,
            'expected_salary_day_of_month': 5,
            'salary_credit_date_m1': '2024-03-05',
            'total_debit_amount_30d': 40000,
            'total_monthly_income': 100000,
            'failed_auto_debits_m1': bounces,
            'failed_auto_debits_m2': 0,
            'lending_app_transaction_count_30d': np.random.randint(0, 10) if seg == 'RISKY' else 0,
            'overdraft_days_30d': np.random.randint(0, 20) if seg == 'RISKY' else 0,
            'end_of_month_balance_m1': savings,
            'end_of_month_balance_m6': savings * 1.5,
            'emi_payment_day_m1': '2024-03-10',
            'emi_payment_day_m2': '2024-02-10',
            'emi_payment_day_m3': '2024-01-10',
            'total_loan_outstanding': emi_inc * 2400000, # 2 year loan roughly
            'total_credit_limit': 500000,
            'employment_category': 'SALARIED',
            'number_of_active_loans': 1 if seg == 'PRIME' else np.random.randint(1, 5),
            'customer_vintage_months': np.random.randint(12, 120),
            'default_flag': default_flag,
            'external_customer_id': raw_ids[i]
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Verify overall default rate
    actual_dr = df['default_flag'].mean()
    print(f"Dataset generated with {actual_dr:.2%} Actual Default Rate.")
    
    output_path = "dataset/portfolio_100k.csv"
    os.makedirs("dataset", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_100k_balanced()
