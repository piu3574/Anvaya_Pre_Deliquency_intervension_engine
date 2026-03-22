import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(num_samples=100000, output_dir="dataset"):
    np.random.seed(42)  # For reproducibility

    print("Generating base demographic features...")
    # 1. Base IDs and Segmentation
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, num_samples + 1)]
    
    # Customer Segments
    # LOW: 60,000 (0 default)
    # MEDIUM: 32,000 (0 default)
    # CRITICAL: 8,000 (1 default)
    segments = ["LOW_STRESS"] * 60000 + ["MEDIUM_STRESS"] * 32000 + ["CRITICAL_STRESS"] * 8000
    np.random.shuffle(segments) # Shuffle so they are distributed across IDs
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_segment": segments
    })
    
    df["default_flag"] = (df["customer_segment"] == "CRITICAL_STRESS").astype(int)

    # Employment Category
    jobs = ["Salaried_Government", "Salaried_Private", "Self_Employed", "Business_Owner", "Freelancer_Gig", "Contractual"]
    probs = [0.20, 0.35, 0.15, 0.10, 0.12, 0.08]
    df["employment_category"] = np.random.choice(jobs, size=num_samples, p=probs)
    
    # Age
    # 18-75, more between 25-55.
    ages = np.floor(np.random.normal(loc=35, scale=12, size=num_samples))
    ages = np.clip(ages, 18, 75).astype(int)
    
    # Slightly older for Gov and Business
    gov_bus_mask = df["employment_category"].isin(["Salaried_Government", "Business_Owner"])
    ages[gov_bus_mask] = np.clip(ages[gov_bus_mask] + 5, 18, 75)
    df["age"] = ages
    
    df["gender"] = np.random.choice(["Male", "Female", "Other"], size=num_samples, p=[0.55, 0.43, 0.02])
    
    # City and Region
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Pune", "Hyderabad", "Ahmedabad", "Jaipur", "Lucknow", "Indore", "Chandigarh"]
    regions = {"Mumbai": "West", "Pune": "West", "Ahmedabad": "West", "Indore": "Central",
               "Delhi": "North", "Jaipur": "North", "Lucknow": "North", "Chandigarh": "North",
               "Bengaluru": "South", "Chennai": "South", "Hyderabad": "South",
               "Kolkata": "East"}
    
    df["city"] = np.random.choice(cities, size=num_samples)
    df["region"] = df["city"].map(regions)

    # Income source mapping
    def map_income_source(emp):
        if emp in ["Salaried_Government", "Salaried_Private"]:
            return np.random.choice(["Fixed_Salary", "Variable_Salary"], p=[0.7, 0.3])
        elif emp in ["Self_Employed", "Business_Owner"]:
            return np.random.choice(["Business_Income", "Mixed"], p=[0.8, 0.2])
        elif emp == "Freelancer_Gig":
            return np.random.choice(["Gig_Income", "Mixed"], p=[0.7, 0.3])
        elif emp == "Contractual":
            return np.random.choice(["Variable_Salary", "Mixed"], p=[0.6, 0.4])
            
    df["income_source_type"] = df["employment_category"].apply(map_income_source)
    df["gig_worker_flag"] = (df["employment_category"] == "Freelancer_Gig").astype(int)

    # Vintage & Cold Start
    df["customer_vintage_months"] = np.random.randint(0, 121, size=num_samples)
    # Increase proportion of short vintage for gig/contractual
    gig_cont_mask = df["employment_category"].isin(["Freelancer_Gig", "Contractual"])
    # Sample down vintage for half of them
    adjust_idx = df.index[gig_cont_mask & (np.random.rand(num_samples) < 0.5)]
    df.loc[adjust_idx, "customer_vintage_months"] = np.random.randint(0, 12, size=len(adjust_idx))
    
    df["cold_start_flag"] = (df["customer_vintage_months"] < 6).astype(int)
    
    # Account open date relative to today (say 2024-03-01)
    today = datetime(2024, 3, 1)
    df["account_open_date"] = df["customer_vintage_months"].apply(lambda x: (today - pd.DateOffset(months=x)).strftime("%Y-%m-%d"))

    print("Generating incomes and limits...")
    # Incomes
    df["monthly_net_salary"] = 0
    sal_mask = df["employment_category"].isin(["Salaried_Government", "Salaried_Private", "Contractual"])
    df.loc[sal_mask, "monthly_net_salary"] = np.random.lognormal(mean=np.log(45000), sigma=0.6, size=sal_mask.sum())
    
    bus_mask = df["employment_category"].isin(["Business_Owner", "Self_Employed"])
    df.loc[bus_mask, "monthly_net_salary"] = 0 # No direct salary, but business income exists
    
    df["total_monthly_income"] = df["monthly_net_salary"].copy()
    # Add non-salary component based on category
    df.loc[bus_mask, "total_monthly_income"] = np.random.lognormal(mean=np.log(80000), sigma=0.7, size=bus_mask.sum())
    gig_mask = df["employment_category"] == "Freelancer_Gig"
    df.loc[gig_mask, "total_monthly_income"] = np.random.lognormal(mean=np.log(30000), sigma=0.8, size=gig_mask.sum())
    
    # Floor income
    df["total_monthly_income"] = np.clip(df["total_monthly_income"], 10000, 1000000).round()
    df["monthly_net_salary"] = np.round(df["monthly_net_salary"])
    
    # Stress tweaks (Lower income for critical stress often, but not strictly required, mostly it's about ratios)

    # Salary Inflow Variance
    variance_map = {"Fixed_Salary": 0.05, "Variable_Salary": 0.2, "Business_Income": 0.3, "Gig_Income": 0.5, "Mixed": 0.25}
    base_var = df["income_source_type"].map(variance_map)
    df["salary_inflow_variance_12m"] = np.clip(np.random.normal(base_var, 0.05), 0.01, 1.0)
    
    # Salary day
    df["expected_salary_day_of_month"] = np.random.randint(1, 6, size=num_samples)
    df.loc[gig_mask, "expected_salary_day_of_month"] = 15 # neutral for gigs
    
    # Recent salary dates (M1..M3)
    for m in [1, 2, 3]:
        delay = np.zeros(num_samples)
        delay[gig_mask] = np.random.randint(-10, 15, size=gig_mask.sum())
        delay[sal_mask] = np.random.randint(0, 3, size=sal_mask.sum()) # Slight delays
        actual_day = np.clip(df["expected_salary_day_of_month"] + delay, 1, 28).astype(int)
        df[f"salary_credit_date_m{m}"] = actual_day.apply(lambda d: f"2023-{12-m+1:02d}-{d:02d}" if 12-m+1 > 0 else f"2023-{(12-m+1)%12 or 12:02d}-{d:02d}")
        
        amt_factor = np.random.normal(1, df["salary_inflow_variance_12m"])
        df[f"salary_credit_amount_m{m}"] = np.clip(df["total_monthly_income"] * amt_factor, 0, None).round()

    print("Generating credit footprints...")
    # Credit Limit
    base_limit_multiplier = {"LOW_STRESS": 3.0, "MEDIUM_STRESS": 2.0, "CRITICAL_STRESS": 1.5}
    mults = df["customer_segment"].map(base_limit_multiplier)
    df["total_credit_limit"] = np.clip(np.round(df["total_monthly_income"] * mults / 10000) * 10000, 20000, 2000000)
    
    df["revolving_credit_outstanding_balance"] = 0.0
    mask_low = df["customer_segment"] == "LOW_STRESS"
    mask_med = df["customer_segment"] == "MEDIUM_STRESS"
    mask_crit = df["customer_segment"] == "CRITICAL_STRESS"
    df.loc[mask_low, "revolving_credit_outstanding_balance"] = df.loc[mask_low, "total_credit_limit"] * np.random.uniform(0.01, 0.2, size=mask_low.sum())
    df.loc[mask_med, "revolving_credit_outstanding_balance"] = df.loc[mask_med, "total_credit_limit"] * np.random.uniform(0.3, 0.7, size=mask_med.sum())
    df.loc[mask_crit, "revolving_credit_outstanding_balance"] = df.loc[mask_crit, "total_credit_limit"] * np.random.uniform(0.8, 1.0, size=mask_crit.sum())
    df["revolving_credit_outstanding_balance"] = df["revolving_credit_outstanding_balance"].round()

    df["has_credit_card"] = np.random.choice([0, 1], p=[0.3, 0.7], size=num_samples)
    df["has_home_loan"] = np.random.choice([0, 1], p=[0.8, 0.2], size=num_samples)
    
    df["number_of_active_loans"] = np.random.randint(1, 4, size=num_samples)
    df.loc[df["customer_segment"] == "CRITICAL_STRESS", "number_of_active_loans"] += np.random.randint(1, 3, size=(df["customer_segment"] == "CRITICAL_STRESS").sum())
    
    # EMIs
    emi_ratio = np.zeros(num_samples)
    emi_ratio[df["customer_segment"] == "LOW_STRESS"] = np.random.uniform(0.1, 0.4, size=(df["customer_segment"] == "LOW_STRESS").sum())
    emi_ratio[df["customer_segment"] == "MEDIUM_STRESS"] = np.random.uniform(0.4, 0.8, size=(df["customer_segment"] == "MEDIUM_STRESS").sum())
    emi_ratio[df["customer_segment"] == "CRITICAL_STRESS"] = np.random.uniform(0.7, 1.2, size=(df["customer_segment"] == "CRITICAL_STRESS").sum())
    
    df["total_monthly_emi_amount"] = (df["total_monthly_income"] * emi_ratio).round()
    df["total_loan_outstanding"] = (df["total_monthly_emi_amount"] * np.random.uniform(12, 48, size=num_samples)).round()
    df["average_loan_interest_rate"] = np.clip(np.random.normal(12, 4, size=num_samples), 5, 30).round(1)
    
    df.loc[gig_mask, "average_loan_interest_rate"] += np.random.uniform(2, 6, size=gig_mask.sum())
    
    # EMI timing
    df["typical_emi_due_day_of_month"] = np.random.randint(1, 15, size=num_samples)
    
    for m in range(1, 7):
        delay = np.zeros(num_samples)
        delay[df["customer_segment"] == "LOW_STRESS"] = np.random.randint(-2, 2, size=(df["customer_segment"] == "LOW_STRESS").sum())
        delay[df["customer_segment"] == "MEDIUM_STRESS"] = np.random.randint(0, 10, size=(df["customer_segment"] == "MEDIUM_STRESS").sum())
        delay[df["customer_segment"] == "CRITICAL_STRESS"] = np.random.randint(5, 25, size=(df["customer_segment"] == "CRITICAL_STRESS").sum())
        
        actual_day = np.clip(df["typical_emi_due_day_of_month"] + delay, 1, 28).astype(int)
        df[f"emi_payment_day_m{m}"] = actual_day.apply(lambda d: f"2023-{12-m+1:02d}-{d:02d}" if 12-m+1 > 0 else f"2023-{(12-m+1)%12 or 12:02d}-{d:02d}")
        
    df["emi_amount_due_last_3m"] = (df["total_monthly_emi_amount"] * 3).round()
    
    paid_ratio = np.ones(num_samples)
    paid_ratio[df["customer_segment"] == "MEDIUM_STRESS"] = np.random.uniform(0.8, 1.0, size=(df["customer_segment"] == "MEDIUM_STRESS").sum())
    paid_ratio[df["customer_segment"] == "CRITICAL_STRESS"] = np.random.uniform(0.2, 0.9, size=(df["customer_segment"] == "CRITICAL_STRESS").sum())
    df["emi_amount_paid_last_3m"] = (df["emi_amount_due_last_3m"] * paid_ratio).round()

    print("Generating Balances and Sequences...")
    # Savings balances
    sv_mult = {"LOW_STRESS": 3.0, "MEDIUM_STRESS": 1.0, "CRITICAL_STRESS": 0.2}
    base_bal = df["total_monthly_income"] * df["customer_segment"].map(sv_mult)
    
    df["current_account_balance"] = (base_bal * np.random.uniform(0.8, 1.2, size=num_samples)).round()
    df["savings_balance_60d_ago"] = (base_bal * np.random.uniform(0.9, 1.5, size=num_samples) * (1 + (df["customer_segment"]=="CRITICAL_STRESS")*1.5)).round()
    
    df["average_daily_balance_30d"] = (df["current_account_balance"] * np.random.uniform(0.9, 1.2, size=num_samples)).round()
    df["minimum_daily_balance_30d"] = (df["average_daily_balance_30d"] * np.random.uniform(0.2, 0.8, size=num_samples)).round()
    df["maximum_daily_balance_30d"] = (df["average_daily_balance_30d"] * np.random.uniform(1.2, 2.0, size=num_samples)).round()
    
    df["overdraft_days_30d"] = 0
    df.loc[df["customer_segment"] == "MEDIUM_STRESS", "overdraft_days_30d"] = np.random.randint(0, 6, size=(df["customer_segment"] == "MEDIUM_STRESS").sum())
    df.loc[df["customer_segment"] == "CRITICAL_STRESS", "overdraft_days_30d"] = np.random.randint(5, 30, size=(df["customer_segment"] == "CRITICAL_STRESS").sum())

    # 12-Month Sequences
    for t in range(1, 13):
        noise = np.random.normal(1, df["salary_inflow_variance_12m"])
        df[f"monthly_net_salary_m{t}"] = (df["monthly_net_salary"] * noise).round()
        df[f"total_monthly_income_m{t}"] = (df["total_monthly_income"] * noise).round()
        
        # EMI generally stable, slightly rising for CRITICAL
        emi_trend = 1.0
        if t <= 6: # further back in time
            df.loc[df["customer_segment"] == "CRITICAL_STRESS", "total_monthly_emi_amount_m"+str(t)] = (df["total_monthly_emi_amount"] * 0.8).round()
            df.loc[df["customer_segment"] != "CRITICAL_STRESS", "total_monthly_emi_amount_m"+str(t)] = df["total_monthly_emi_amount"].copy()
        else:
            df[f"total_monthly_emi_amount_m{t}"] = df["total_monthly_emi_amount"].copy()
            
        # Balances trending
        if "LOW_STRESS" in df["customer_segment"].values:
            low_m = df["customer_segment"] == "LOW_STRESS"
            df.loc[low_m, f"end_of_month_balance_m{t}"] = (df.loc[low_m, "current_account_balance"] * np.random.uniform(0.9, 1.1, size=low_m.sum())).round()
        
        med_m = df["customer_segment"] == "MEDIUM_STRESS"
        df.loc[med_m, f"end_of_month_balance_m{t}"] = (df.loc[med_m, "current_account_balance"] * (1 + (12-t)*0.05) * np.random.uniform(0.9, 1.1, size=med_m.sum())).round()
        
        crit_m = df["customer_segment"] == "CRITICAL_STRESS"
        df.loc[crit_m, f"end_of_month_balance_m{t}"] = (df.loc[crit_m, "current_account_balance"] * (1 + (12-t)*0.2) * np.random.uniform(0.9, 1.1, size=crit_m.sum())).round()
        
        # Override negative balances safely
        df[f"end_of_month_balance_m{t}"] = np.clip(df[f"end_of_month_balance_m{t}"], 0, None)
        
        # Failed debits
        df[f"failed_auto_debits_m{t}"] = 0
        df.loc[med_m, f"failed_auto_debits_m{t}"] = np.random.poisson((t/12)*1, size=med_m.sum())
        df.loc[crit_m, f"failed_auto_debits_m{t}"] = np.random.poisson((t/12)*4, size=crit_m.sum())
        
        # Overdraft days
        df[f"overdraft_days_m{t}"] = 0
        df.loc[med_m, f"overdraft_days_m{t}"] = np.clip(np.random.poisson((t/12)*2, size=med_m.sum()), 0, 30)
        df.loc[crit_m, f"overdraft_days_m{t}"] = np.clip(np.random.poisson((t/12)*15, size=crit_m.sum()), 0, 30)

    print("Generating Spends...")
    df["total_transaction_count_30d"] = np.random.randint(10, 150, size=num_samples)
    
    spend_mult = {"LOW_STRESS": 0.4, "MEDIUM_STRESS": 0.6, "CRITICAL_STRESS": 0.9} # Spend as portion of income
    base_spends = df["total_monthly_income"] * df["customer_segment"].map(spend_mult)
    df["total_debit_amount_30d"] = (base_spends * np.random.uniform(0.8, 1.2, size=num_samples)).round()
    df["total_credit_amount_30d"] = (df["total_monthly_income"] * np.random.uniform(0.9, 1.5, size=num_samples)).round()
    
    df["atm_withdrawal_count_30d"] = np.random.randint(0, 10, size=num_samples)
    df.loc[df["employment_category"].isin(["Self_Employed", "Freelancer_Gig"]), "atm_withdrawal_count_30d"] += np.random.randint(2, 8, size=df["employment_category"].isin(["Self_Employed", "Freelancer_Gig"]).sum())
    df["atm_withdrawal_amount_30d"] = (df["atm_withdrawal_count_30d"] * np.random.uniform(1000, 5000, size=num_samples)).round()
    
    # Categories
    grocery_shares = np.where(df["customer_segment"] == "CRITICAL_STRESS", np.random.uniform(0.4, 0.7, size=num_samples), np.random.uniform(0.1, 0.3, size=num_samples))
    dining_shares = np.where(df["customer_segment"] == "LOW_STRESS", np.random.uniform(0.1, 0.3, size=num_samples), np.random.uniform(0.01, 0.05, size=num_samples))
    ent_shares = np.where(df["customer_segment"] == "LOW_STRESS", np.random.uniform(0.1, 0.2, size=num_samples), np.random.uniform(0.01, 0.05, size=num_samples))
    
    df["grocery_spend_amount_30d"] = (df["total_debit_amount_30d"] * grocery_shares).round()
    df["dining_spend_amount_30d"] = (df["total_debit_amount_30d"] * dining_shares).round()
    df["entertainment_spend_amount_30d"] = (df["total_debit_amount_30d"] * ent_shares).round()
    df["other_retail_spend_amount_30d"] = (df["total_debit_amount_30d"] * (1 - grocery_shares - dining_shares - ent_shares) * 0.5).round() # Ensures sum <= total
    
    df["grocery_spend_amount_90d"] = (df["grocery_spend_amount_30d"] * np.random.uniform(2.5, 3.5, size=num_samples)).round()
    df["dining_spend_amount_90d"] = (df["dining_spend_amount_30d"] * np.random.uniform(2.5, 3.5, size=num_samples)).round()
    df["entertainment_spend_amount_90d"] = (df["entertainment_spend_amount_30d"] * np.random.uniform(2.5, 3.5, size=num_samples)).round()
    df["other_retail_spend_amount_90d"] = (df["other_retail_spend_amount_30d"] * np.random.uniform(2.5, 3.5, size=num_samples)).round()
    
    df["lending_app_transaction_count_30d"] = 0
    df.loc[med_m, "lending_app_transaction_count_30d"] = np.random.poisson(1, size=med_m.sum())
    df.loc[crit_m, "lending_app_transaction_count_30d"] = np.random.poisson(4, size=crit_m.sum())
    
    df["auto_debit_total_count_30d"] = np.random.randint(2, 10, size=num_samples)
    df["auto_debit_failed_insufficient_funds_30d"] = 0
    df.loc[med_m, "auto_debit_failed_insufficient_funds_30d"] = np.random.poisson(0.5, size=med_m.sum())
    df.loc[crit_m, "auto_debit_failed_insufficient_funds_30d"] = np.random.poisson(2, size=crit_m.sum())
    df["auto_debit_failed_other_reasons_30d"] = np.random.poisson(0.1, size=num_samples)
    
    # Cap failed so it doesn't exceed total
    total_fails = df["auto_debit_failed_insufficient_funds_30d"] + df["auto_debit_failed_other_reasons_30d"]
    df["auto_debit_total_count_30d"] = np.maximum(df["auto_debit_total_count_30d"], total_fails + np.random.randint(1, 5, size=num_samples))
    df["auto_debit_success_count_30d"] = df["auto_debit_total_count_30d"] - total_fails

    # Credits breakdown
    df["salary_credit_amount_30d"] = 0.0
    df.loc[sal_mask, "salary_credit_amount_30d"] = (df.loc[sal_mask, "total_credit_amount_30d"] * 0.8).round()
    
    df["business_income_credit_amount_30d"] = 0.0
    df.loc[bus_mask, "business_income_credit_amount_30d"] = (df.loc[bus_mask, "total_credit_amount_30d"] * 0.7).round()
    
    df["gig_income_credit_amount_30d"] = 0.0
    df.loc[gig_mask, "gig_income_credit_amount_30d"] = (df.loc[gig_mask, "total_credit_amount_30d"] * 0.6).round()
    
    df["loan_disbursement_credit_amount_30d"] = 0.0
    df.loc[crit_m, "loan_disbursement_credit_amount_30d"] = (df.loc[crit_m, "total_credit_amount_30d"] * 0.2).round()
    
    df["own_account_transfer_credit_amount_30d"] = (df["total_credit_amount_30d"] * 0.05).round()
    df["other_credit_amount_30d"] = (df["total_credit_amount_30d"] - df[["salary_credit_amount_30d", "business_income_credit_amount_30d", "gig_income_credit_amount_30d", "loan_disbursement_credit_amount_30d", "own_account_transfer_credit_amount_30d"]].sum(axis=1)).clip(lower=0).round()
    
    # Governance
    df["vulnerability_flag"] = np.random.choice([0, 1], p=[0.95, 0.05], size=num_samples)
    
    intervention_dates = []
    for c in df["customer_segment"]:
        if c == "CRITICAL_STRESS":
            intervention_dates.append((today - pd.DateOffset(days=np.random.randint(1, 30))).strftime("%Y-%m-%d"))
        elif c == "MEDIUM_STRESS":
            intervention_dates.append((today - pd.DateOffset(days=np.random.randint(30, 180))).strftime("%Y-%m-%d"))
        else:
            if np.random.rand() < 0.1:
                intervention_dates.append((today - pd.DateOffset(days=np.random.randint(180, 500))).strftime("%Y-%m-%d"))
            else:
                intervention_dates.append("1900-01-01") # Dummy for none
    df["last_intervention_date"] = intervention_dates
    
    df["intervention_pending_flag"] = ((df["customer_segment"] == "CRITICAL_STRESS") & (np.random.rand(num_samples) < 0.3)).astype(int)
    df["onboarding_aa_consent_flag"] = np.random.choice([0, 1], p=[0.1, 0.9], size=num_samples)
    df["feature_store_schema_version"] = "v1.0"

    print("Checking constraints and saving...")
    # Fill remaining NaNs to comply with 'no missing values'
    df.fillna(0, inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "gypti_bank_synthetic_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Data generation complete! Saved to {out_path}")
    print(df["customer_segment"].value_counts())
    print(df["default_flag"].value_counts())

if __name__ == "__main__":
    generate_synthetic_data()
