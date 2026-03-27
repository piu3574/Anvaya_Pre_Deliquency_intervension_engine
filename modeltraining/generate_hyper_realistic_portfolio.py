import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set seed for reproducibility
np.random.seed(42)

def generate_realistic_dataset(n_rows=100000):
    print(f"🚀 Generating {n_rows} rows of hyper-realistic banking data...")
    
    # 1. Identifiers
    customer_ids = [f"CUST_{i+1:06d}" for i in range(n_rows)]
    
    # 2. Demographics
    ages = np.random.normal(38, 12, n_rows).clip(21, 75).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_rows, p=[0.55, 0.43, 0.02])
    employment_cats = np.random.choice(
        ['salaried', 'self_employed', 'business_owner', 'retired', 'student'], 
        n_rows, p=[0.65, 0.15, 0.10, 0.07, 0.03]
    )
    sectors = np.random.choice(
        ['IT', 'Manufacturing', 'Government', 'Finance', 'Retail', 'Gig', 'Agriculture', 'Healthcare', 'Education', 'Other'],
        n_rows, p=[0.15, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05, 0.10, 0.10, 0.05]
    )
    regions = np.random.choice(['Metro', 'Urban', 'Semi-Urban', 'Rural'], n_rows, p=[0.3, 0.4, 0.2, 0.1])

    # 3. Relationship
    vintages = (np.random.exponential(48, n_rows) + 6).clip(6, 240).astype(int)
    base_date = datetime(2024, 3, 1)
    open_dates = [(base_date - timedelta(days=int(v*30.41))).strftime('%Y-%m-%d') for v in vintages]

    # 4. Income & Credits (Base Salaries per employment type)
    base_salaries = {
        'salaried': 45000, 'self_employed': 35000, 'business_owner': 85000, 
        'retired': 25000, 'student': 5000
    }
    salaries = np.array([np.random.lognormal(np.log(base_salaries[c]), 0.4) for c in employment_cats])
    
    total_salary_credit_30d = salaries
    non_salary_credits = np.random.gamma(2, 5000, n_rows) 
    total_credits_30d = total_salary_credit_30d + non_salary_credits
    
    expected_salary_date = np.random.choice([1, 5, 7, 10, 15], n_rows, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    # Delays sampled for logic F3 later
    salary_delays = np.random.choice([0, 1, 2, 3, 5, 10, 20], n_rows, p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
    salary_credit_date_m1 = (expected_salary_date + salary_delays).clip(1, 31)
    
    avg_monthly_credits_3m = total_credits_30d * np.random.uniform(0.9, 1.1, n_rows)
    income_volatility = np.random.beta(2, 8, n_rows) 
    salary_inconsistency_flag = (salary_delays > 3).astype(int)

    # 5. Debt & Loans
    # DTI (Debt to Income) usually 0.2 to 0.6
    dti = np.random.beta(2, 5, n_rows) * 0.8
    total_monthly_emi = (total_salary_credit_30d * dti).clip(0, None)
    
    num_loans = np.random.poisson(1.5, n_rows).clip(0, 10)
    total_limit = total_salary_credit_30d * np.random.uniform(5, 50, n_rows)
    total_outstanding = total_limit * np.random.uniform(0.1, 0.9, n_rows) * (dti * 2).clip(0, 1)
    
    is_new_to_credit = (num_loans == 0).astype(int)
    max_overdue_12m = (np.random.exponential(2, n_rows) * (dti * 10)).clip(0, 90).astype(int)
    default_hist_12m = (max_overdue_12m // 30).clip(0, 5)
    cc_count = np.random.poisson(1.2, n_rows).clip(0, 8)
    avg_int_rate = np.random.uniform(8, 24, n_rows) + (dti * 12)

    # 6. Spending
    total_debits = total_credits_30d * np.random.uniform(0.7, 1.3, n_rows)
    total_spend = total_debits * np.random.uniform(0.4, 0.8, n_rows)
    atm_withdrawals = np.random.gamma(2, 2000, n_rows).clip(0, 50000)
    num_txns = np.random.poisson(45, n_rows).clip(5, 300)
    high_val_txns = np.random.poisson(1.5, n_rows).clip(0, 15)

    # 7. Savings & Liquidity
    savings_m2 = np.random.lognormal(11, 1.5, n_rows).clip(100, 10000000)
    savings_current = savings_m2 * np.random.uniform(0.5, 1.2, n_rows)
    avg_mon_end_bal = (savings_m2 + savings_current) / 2 * np.random.uniform(0.9, 1.1, n_rows)
    
    overdraft_days = (np.random.exponential(1, n_rows) * (dti * 5)).clip(0, 30).astype(int)
    auto_debit_fails = np.random.poisson(0.2 + (dti * 2), n_rows).clip(0, 10)
    lending_app_txns = np.random.poisson(0.1 + (dti * 5), n_rows).clip(0, 25)
    emi_pay_day = (expected_salary_date + np.random.randint(0, 5, n_rows)).clip(1, 31)

    # 8. RISK SCORING ENGINE (Hidden Logic)
    # We create a composite score based on stressors
    stress_f1 = (total_monthly_emi / (total_salary_credit_30d + 1)).clip(0, 1) # EMI/Income
    stress_f2 = (1 - (savings_current / (savings_m2 + 1))).clip(0, 1) # Savings drawdown
    stress_f3 = (salary_delays / 10).clip(0, 1) # Delay
    stress_f5 = (auto_debit_fails / 5).clip(0, 1) # Fails
    stress_f6 = (lending_app_txns / 10).clip(0, 1) # Apps
    stress_f14 = (total_outstanding / (total_limit + 1)).clip(0, 1) # Utilization
    
    # Combined Risk Score (0 to 1)
    risk_score = (
        0.25 * stress_f1 + 
        0.15 * stress_f2 + 
        0.10 * stress_f3 + 
        0.20 * stress_f5 + 
        0.10 * stress_f6 + 
        0.20 * stress_f14
    )
    
    # Calibration to targets: 
    # Green (~60%): risk < 0.35, Yellow (~30%): 0.35-0.65, Red (~10%): > 0.65
    risk_band = []
    default_status = []
    
    for score in risk_score:
        if score < 0.35: # GREEN
            band = 'GREEN'
            prob_default = np.random.uniform(0.005, 0.05) # Realistic low PD
        elif score < 0.65: # YELLOW
            band = 'YELLOW'
            prob_default = np.random.uniform(0.15, 0.45)
        else: # RED
            band = 'RED'
            prob_default = np.random.uniform(0.55, 0.95)
            
        risk_band.append(band)
        default_status.append(1 if np.random.random() < prob_default else 0)

    # Final Adjustment to hit ~6% exact default rate if needed (but probabilistic is better)
    # Current simulation will yield ~6-8% naturally based on the weights above.
    
    # Flatten all vectors for safe DF creation
    cols_to_check = {
        'customer_id': list(customer_ids), 'age': np.array(ages).flatten(), 'gender': np.array(genders).flatten(), 
        'employment_category': np.array(employment_cats).flatten(), 'employment_sector': np.array(sectors).flatten(), 
        'region': np.array(regions).flatten(), 'customer_vintage_months': np.array(vintages).flatten(), 
        'account_open_date': list(open_dates), 'max_overdue_days_12m': np.array(max_overdue_12m).flatten(), 
        'default_history_12m': np.array(default_hist_12m).flatten(), 'existing_credit_cards_count': np.array(cc_count).flatten(), 
        'is_new_to_credit_flag': np.array(is_new_to_credit).flatten(), 'total_salary_credit_30d': np.array(total_salary_credit_30d).flatten(), 
        'total_credits_30d': np.array(total_credits_30d).flatten(), 'non_salary_credits_30d': np.array(non_salary_credits).flatten(), 
        'salary_credit_date_m1': np.array(salary_credit_date_m1).flatten(), 'expected_salary_date': np.array(expected_salary_date).flatten(), 
        'avg_monthly_credits_3m': np.array(avg_monthly_credits_3m).flatten(), 'income_volatility_ratio_3m': np.array(income_volatility).flatten(), 
        'salary_inconsistency_flag': np.array(salary_inconsistency_flag).flatten(), 'total_monthly_emi_amount': np.array(total_monthly_emi).flatten(), 
        'number_of_active_loans': np.array(num_loans).flatten(), 'total_loan_outstanding': np.array(total_outstanding).flatten(), 
        'total_credit_limit': np.array(total_limit).flatten(), 'avg_interest_rate_loans': np.array(avg_int_rate).flatten(), 
        'total_debit_amount_30d': np.array(total_debits).flatten(), 'total_spend_30d': np.array(total_spend).flatten(), 
        'atm_cash_withdrawals_30d': np.array(atm_withdrawals).flatten(), 'num_transactions_30d': np.array(num_txns).flatten(), 
        'high_value_txn_count_30d': np.array(high_val_txns).flatten(), 'savings_balance_60d_ago': np.array(savings_m2).flatten(), 
        'savings_balance_current': np.array(savings_current).flatten(), 'avg_month_end_balance_3m': np.array(avg_mon_end_bal).flatten(), 
        'overdraft_days_30d': np.array(overdraft_days).flatten(), 'auto_debit_failure_count_30d': np.array(auto_debit_fails).flatten(), 
        'lending_app_transaction_count_30d': np.array(lending_app_txns).flatten(), 'emi_payment_day_m1': np.array(emi_pay_day).flatten(), 
        'default_status': list(default_status), 'risk_band': list(risk_band)
    }
    
    for k, v in cols_to_check.items():
        v_len = len(v)
        v_type = type(v)
        if v_len != n_rows:
            print(f"❌ LENGTH MISMATCH: {k} has {v_len} rows (Type: {v_type}), expected {n_rows}")
        else:
            # Check for multi-dimensionality (e.g. (100000, 1) instead of (100000,))
            if hasattr(v, 'shape') and len(v.shape) > 1 and v.shape[1] > 1:
                print(f"❌ SHAPE MISMATCH: {k} has shape {v.shape}")

    # Construct DataFrame
    try:
        df = pd.DataFrame(cols_to_check)
    except Exception as e:
        print(f"❌ DATAFRAME FAILURE: {e}")
        # Print shape of everything again
        for k, v in cols_to_check.items():
            print(f"  {k}: {np.shape(v)}")
        raise e

    # Stats Print
    print(f"✅ Generation Complete.")
    print(f"Default Rate: {df.default_status.mean()*100:.2f}%")
    print(f"Bands:\n{df.risk_band.value_counts(normalize=True)*100}")
    
    output_path = "dataset/hyper_realistic_portfolio_100k.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_realistic_dataset()
