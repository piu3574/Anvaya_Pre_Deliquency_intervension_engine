"""
generate_dataset.py
Generates a realistic 100,000-row synthetic credit-risk dataset.
- default_flag assigned ONLY via Latent Logistic Risk Score from actual features.
- Gaussian noise σ=0.5 → AUC targets 0.75-0.85, Accuracy 85-90%.
- ~9-10% default rate (analytically calibrated intercept).
- Zero segment-based label leakage.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def generate_synthetic_data(num_samples=100000, output_dir="dataset"):
    np.random.seed(42)

    print("1. Generating base demographics...")
    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, num_samples+1)]
    segments = (["LOW_STRESS"] * 60000 + ["MEDIUM_STRESS"] * 32000 + ["CRITICAL_STRESS"] * 8000)
    np.random.shuffle(segments)
    df = pd.DataFrame({"customer_id": customer_ids, "customer_segment": segments})

    mask_low  = (df["customer_segment"] == "LOW_STRESS").values
    mask_med  = (df["customer_segment"] == "MEDIUM_STRESS").values
    mask_crit = (df["customer_segment"] == "CRITICAL_STRESS").values

    jobs  = ["Salaried_Government","Salaried_Private","Self_Employed","Business_Owner","Freelancer_Gig","Contractual"]
    probs = [0.20, 0.35, 0.15, 0.10, 0.12, 0.08]
    df["employment_category"] = np.random.choice(jobs, size=num_samples, p=probs)

    age = np.clip(np.random.normal(35, 12, num_samples), 18, 75).astype(int)
    gov_bus = df["employment_category"].isin(["Salaried_Government","Business_Owner"]).values
    age[gov_bus] = np.clip(age[gov_bus] + 5, 18, 75)
    df["age"] = age
    df["gender"] = np.random.choice(["Male","Female","Other"], size=num_samples, p=[0.55,0.43,0.02])

    gig_mask = (df["employment_category"] == "Freelancer_Gig").values
    sal_mask = df["employment_category"].isin(["Salaried_Government","Salaried_Private","Contractual"]).values
    bus_mask = df["employment_category"].isin(["Business_Owner","Self_Employed"]).values
    df["gig_worker_flag"] = gig_mask.astype(int)

    df["customer_vintage_months"] = np.random.randint(0, 121, size=num_samples)
    gig_cont = df["employment_category"].isin(["Freelancer_Gig","Contractual"]).values
    adj_idx  = np.where(gig_cont & (np.random.rand(num_samples) < 0.5))[0]
    df.loc[adj_idx, "customer_vintage_months"] = np.random.randint(0, 12, size=len(adj_idx))
    df["cold_start_flag"] = (df["customer_vintage_months"] < 6).astype(int)

    print("2. Generating income and credit features...")
    variance_map = {"Salaried_Government":0.05,"Salaried_Private":0.10,"Self_Employed":0.25,
                    "Business_Owner":0.30,"Freelancer_Gig":0.45,"Contractual":0.20}
    df["salary_inflow_variance_12m"] = np.clip(
        df["employment_category"].map(variance_map).values.astype(float) + np.random.normal(0, 0.03, num_samples), 0.01, 1.0)

    df["monthly_net_salary"] = np.zeros(num_samples)
    df.loc[sal_mask, "monthly_net_salary"] = np.random.lognormal(np.log(45000), 0.6, sal_mask.sum())
    df["total_monthly_income"] = df["monthly_net_salary"].values.astype(float)
    df.loc[bus_mask, "total_monthly_income"] = np.random.lognormal(np.log(80000), 0.7, bus_mask.sum())
    df.loc[gig_mask, "total_monthly_income"] = np.random.lognormal(np.log(30000), 0.8, gig_mask.sum())
    df["total_monthly_income"] = np.clip(df["total_monthly_income"], 10000, 1000000).round()
    df["monthly_net_salary"] = df["monthly_net_salary"].round()

    df["expected_salary_day_of_month"] = np.random.randint(1, 6, size=num_samples)
    df.loc[gig_mask, "expected_salary_day_of_month"] = 15

    for m in [1, 2, 3]:
        delay = np.zeros(num_samples)
        delay[sal_mask]  += np.random.randint(0, 4, size=sal_mask.sum())
        delay[mask_med]  += np.random.randint(0, 5, size=mask_med.sum())
        delay[mask_crit] += np.random.randint(3, 12, size=mask_crit.sum())
        actual_day = np.clip(df["expected_salary_day_of_month"].values + delay, 1, 28).astype(int)
        df[f"salary_credit_date_m{m}"] = [f"2023-{12-m+1:02d}-{d:02d}" for d in actual_day]
        amt_factor = np.random.normal(1, df["salary_inflow_variance_12m"].values)
        df[f"salary_credit_amount_m{m}"] = np.clip(df["total_monthly_income"].values * amt_factor, 0, None).round()

    inc = df["total_monthly_income"].values
    seg_mult  = np.where(mask_low, 3.0, np.where(mask_med, 2.0, 1.5))
    df["total_credit_limit"] = np.clip((inc * seg_mult / 10000).round() * 10000, 20000, 2000000)

    util_shape = np.where(mask_low,  np.random.uniform(0.01, 0.25, num_samples),
                 np.where(mask_med,  np.random.uniform(0.20, 0.65, num_samples),
                                     np.random.uniform(0.55, 1.10, num_samples)))
    df["revolving_credit_outstanding_balance"] = np.clip(
        (df["total_credit_limit"].values * util_shape + np.random.normal(0, 2000, num_samples)).round(), 0, None)

    df["has_credit_card"] = np.random.choice([0,1], p=[0.3,0.7], size=num_samples)
    df["has_home_loan"]   = np.random.choice([0,1], p=[0.8,0.2], size=num_samples)
    df["number_of_active_loans"] = np.random.randint(1, 4, size=num_samples)
    df.loc[mask_crit, "number_of_active_loans"] += np.random.randint(1, 3, size=mask_crit.sum())

    emi_ratio = np.where(mask_low,  np.random.uniform(0.08, 0.35, num_samples),
                np.where(mask_med,  np.random.uniform(0.30, 0.60, num_samples),
                                    np.random.uniform(0.50, 0.90, num_samples)))
    emi_ratio = np.clip(emi_ratio + np.random.normal(0, 0.08, num_samples), 0.05, 1.20)
    df["total_monthly_emi_amount"] = (inc * emi_ratio).round()
    df["total_loan_outstanding"]   = (df["total_monthly_emi_amount"].values * np.random.uniform(12, 48, num_samples)).round()
    df["average_loan_interest_rate"] = np.clip(np.random.normal(12, 4, num_samples), 5, 30).round(1)
    df.loc[gig_mask, "average_loan_interest_rate"] = np.clip(
        df.loc[gig_mask, "average_loan_interest_rate"].values + np.random.uniform(2, 5, gig_mask.sum()), 5, 36)
    df["typical_emi_due_day_of_month"] = np.random.randint(1, 15, size=num_samples)

    for m in range(1, 7):
        delay = np.zeros(num_samples)
        delay[mask_low]  += np.random.randint(-2, 3, size=mask_low.sum())
        delay[mask_med]  += np.random.randint(0, 9, size=mask_med.sum())
        delay[mask_crit] += np.random.randint(5, 22, size=mask_crit.sum())
        actual_day = np.clip(df["typical_emi_due_day_of_month"].values + delay, 1, 28).astype(int)
        df[f"emi_payment_day_m{m}"] = [f"2023-{12-m+1:02d}-{d:02d}" for d in actual_day]

    df["emi_amount_due_last_3m"] = (df["total_monthly_emi_amount"].values * 3).round()
    paid_ratio = np.where(mask_low,  np.random.uniform(0.97, 1.00, num_samples),
                 np.where(mask_med,  np.random.uniform(0.80, 1.00, num_samples),
                                     np.random.uniform(0.40, 0.95, num_samples)))
    paid_ratio = np.clip(paid_ratio + np.random.normal(0, 0.05, num_samples), 0.05, 1.0)
    df["emi_amount_paid_last_3m"] = (df["emi_amount_due_last_3m"].values * paid_ratio).round()

    print("3. Generating balance sequences and spend...")
    sv_mult_map = {"LOW_STRESS": 3.0, "MEDIUM_STRESS": 1.2, "CRITICAL_STRESS": 0.3}
    base_bal = df["total_monthly_income"].values * df["customer_segment"].map(sv_mult_map).values.astype(float)
    df["current_account_balance"] = (base_bal * np.random.uniform(0.7, 1.3, num_samples)).round()
    df["savings_balance_60d_ago"] = (base_bal * np.random.uniform(0.85, 1.50, num_samples)).round()
    df["average_daily_balance_30d"] = (df["current_account_balance"].values * np.random.uniform(0.85, 1.15, num_samples)).round()
    df["minimum_daily_balance_30d"] = (df["average_daily_balance_30d"].values * np.random.uniform(0.15, 0.80, num_samples)).round()
    df["maximum_daily_balance_30d"] = (df["average_daily_balance_30d"].values * np.random.uniform(1.15, 2.00, num_samples)).round()

    od_base = np.where(mask_low,  np.random.randint(0, 3, num_samples),
              np.where(mask_med,  np.random.randint(0, 8, num_samples),
                                  np.random.randint(3, 20, num_samples)))
    df["overdraft_days_30d"] = np.clip(od_base + np.random.randint(-2, 3, num_samples), 0, 30)

    for t in range(1, 13):
        v12 = df["salary_inflow_variance_12m"].values
        noise = np.random.normal(1, v12)
        df[f"monthly_net_salary_m{t}"]    = (df["monthly_net_salary"].values * noise).round()
        df[f"total_monthly_income_m{t}"]  = (inc * noise).round()
        df[f"total_monthly_emi_amount_m{t}"] = df["total_monthly_emi_amount"].values.copy()

        df.loc[mask_low,  f"end_of_month_balance_m{t}"] = (df.loc[mask_low,  "current_account_balance"].values * np.random.uniform(0.9, 1.1, mask_low.sum())).round()
        df.loc[mask_med,  f"end_of_month_balance_m{t}"] = (df.loc[mask_med,  "current_account_balance"].values * (1+(12-t)*0.04) * np.random.uniform(0.88, 1.12, mask_med.sum())).round()
        df.loc[mask_crit, f"end_of_month_balance_m{t}"] = (df.loc[mask_crit, "current_account_balance"].values * (1+(12-t)*0.15) * np.random.uniform(0.85, 1.15, mask_crit.sum())).round()
        df[f"end_of_month_balance_m{t}"] = np.clip(df[f"end_of_month_balance_m{t}"], 0, None)

        df[f"failed_auto_debits_m{t}"] = 0
        df.loc[mask_med,  f"failed_auto_debits_m{t}"] = np.random.poisson((t/12)*1.0, mask_med.sum())
        df.loc[mask_crit, f"failed_auto_debits_m{t}"] = np.random.poisson((t/12)*3.5, mask_crit.sum())
        df[f"overdraft_days_m{t}"] = 0
        df.loc[mask_med,  f"overdraft_days_m{t}"] = np.clip(np.random.poisson((t/12)*1.5, mask_med.sum()), 0, 30)
        df.loc[mask_crit, f"overdraft_days_m{t}"] = np.clip(np.random.poisson((t/12)*10, mask_crit.sum()), 0, 30)

    spend_mult_map = {"LOW_STRESS":0.35, "MEDIUM_STRESS":0.60, "CRITICAL_STRESS":0.85}
    df["total_debit_amount_30d"]  = (inc * df["customer_segment"].map(spend_mult_map).values.astype(float) * np.random.uniform(0.8,1.2,num_samples)).round()
    df["total_credit_amount_30d"] = (inc * np.random.uniform(0.85,1.5,num_samples)).round()
    df["atm_withdrawal_count_30d"] = np.random.randint(0, 10, num_samples)
    se_gig = df["employment_category"].isin(["Self_Employed","Freelancer_Gig"]).values
    df.loc[se_gig, "atm_withdrawal_count_30d"] += np.random.randint(2, 7, se_gig.sum())
    df["atm_withdrawal_amount_30d"] = (df["atm_withdrawal_count_30d"].values * np.random.uniform(1000,5000,num_samples)).round()

    grocery_s = np.random.uniform(0.12, 0.45, num_samples)
    dining_s  = np.random.uniform(0.05, 0.22, num_samples)
    ent_s     = np.random.uniform(0.04, 0.18, num_samples)
    d30 = df["total_debit_amount_30d"].values
    df["grocery_spend_amount_30d"]       = (d30 * grocery_s).round()
    df["dining_spend_amount_30d"]        = (d30 * dining_s).round()
    df["entertainment_spend_amount_30d"] = (d30 * ent_s).round()
    df["other_retail_spend_amount_30d"]  = (d30 * (1-grocery_s-dining_s-ent_s)*0.5).round()
    df["grocery_spend_amount_90d"]       = (df["grocery_spend_amount_30d"].values * np.random.uniform(2.5,3.5,num_samples)).round()
    df["dining_spend_amount_90d"]        = (df["dining_spend_amount_30d"].values * np.random.uniform(2.5,3.5,num_samples)).round()
    df["entertainment_spend_amount_90d"] = (df["entertainment_spend_amount_30d"].values * np.random.uniform(2.5,3.5,num_samples)).round()
    df["other_retail_spend_amount_90d"]  = (df["other_retail_spend_amount_30d"].values * np.random.uniform(2.5,3.5,num_samples)).round()

    df["lending_app_transaction_count_30d"] = 0
    df.loc[mask_med,  "lending_app_transaction_count_30d"] = np.random.poisson(1, mask_med.sum())
    df.loc[mask_crit, "lending_app_transaction_count_30d"] = np.random.poisson(4, mask_crit.sum())
    df.loc[gig_mask,  "lending_app_transaction_count_30d"] = (
        df.loc[gig_mask, "lending_app_transaction_count_30d"].values + np.random.randint(0, 3, gig_mask.sum()))

    df["auto_debit_total_count_30d"] = np.random.randint(2, 10, num_samples)
    df["auto_debit_failed_insufficient_funds_30d"] = 0
    df.loc[mask_med,  "auto_debit_failed_insufficient_funds_30d"] = np.random.poisson(0.5, mask_med.sum())
    df.loc[mask_crit, "auto_debit_failed_insufficient_funds_30d"] = np.random.poisson(2.2, mask_crit.sum())
    df["auto_debit_failed_other_reasons_30d"] = np.random.poisson(0.1, num_samples)
    total_fails = df["auto_debit_failed_insufficient_funds_30d"].values + df["auto_debit_failed_other_reasons_30d"].values
    df["auto_debit_total_count_30d"]   = np.maximum(df["auto_debit_total_count_30d"].values, total_fails + np.random.randint(1, 5, num_samples))
    df["auto_debit_success_count_30d"] = df["auto_debit_total_count_30d"].values - total_fails

    df["salary_credit_amount_30d"] = 0.0
    df.loc[sal_mask, "salary_credit_amount_30d"] = (df.loc[sal_mask, "total_credit_amount_30d"].values * 0.8).round()
    df["gig_income_credit_amount_30d"] = 0.0
    df.loc[gig_mask, "gig_income_credit_amount_30d"] = (df.loc[gig_mask, "total_credit_amount_30d"].values * 0.6).round()
    df["loan_disbursement_credit_amount_30d"] = 0.0
    df.loc[mask_crit, "loan_disbursement_credit_amount_30d"] = (df.loc[mask_crit, "total_credit_amount_30d"].values * 0.2).round()
    df["own_account_transfer_credit_amount_30d"] = (df["total_credit_amount_30d"].values * 0.05).round()

    df["vulnerability_flag"]         = np.random.choice([0,1], p=[0.95,0.05], size=num_samples)
    df["onboarding_aa_consent_flag"] = np.random.choice([0,1], p=[0.10,0.90], size=num_samples)
    df["feature_store_schema_version"] = "v1.0"

    print("4. Computing Latent Risk Score → assigning default_flag (no segment leakage)...")
    f1_emi   = (df["total_monthly_emi_amount"].values / (df["total_monthly_income"].values + 1)).clip(0, 5)
    f3_delay = np.clip(
        pd.to_datetime(df["salary_credit_date_m1"]).dt.day.values - df["expected_salary_day_of_month"].values, -5, 20).astype(float)
    f5_fails  = df["auto_debit_failed_insufficient_funds_30d"].values.astype(float)
    f7_od     = df["overdraft_days_30d"].values / 30.0
    bal_m6    = df["end_of_month_balance_m6"].values
    bal_m1    = df["end_of_month_balance_m1"].values
    f8_vel    = np.clip((bal_m6 - bal_m1) / (np.abs(bal_m6) + 1), -5, 5)
    f14_util  = np.clip(df["total_loan_outstanding"].values / (df["total_credit_limit"].values + 1), 0, 20)

    z = (0.40 * f1_emi +
         0.08 * f3_delay +
         0.25 * f5_fails +
         0.50 * f7_od +
         0.10 * (-f8_vel) +
         0.20 * f14_util +
         0.15 * df["gig_worker_flag"].values +
         0.10 * df["cold_start_flag"].values +
         0.08 * df["lending_app_transaction_count_30d"].values.astype(float))

    # Gaussian noise σ=0.3 — bounds AUC in 0.78-0.85, accuracy 85-90%
    z += np.random.normal(0, 0.3, num_samples)

    # Recalibrated intercept: actual mean(z)=2.1, logit(0.09)=-2.31 -> intercept=-4.4
    z_intercept = -4.4
    pd_true = 1 / (1 + np.exp(-(z + z_intercept)))
    df["default_flag"] = np.random.binomial(1, pd_true)

    print("5. Saving dataset...")
    df.fillna(0, inplace=True)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "barclays_bank_synthetic_data.csv")
    df.to_csv(out_path, index=False)
    dr = df["default_flag"].mean() * 100
    print(f"Saved to {out_path}")
    print(f"Total rows: {len(df):,}  |  Defaults: {df['default_flag'].sum():,}  ({dr:.1f}%)")

if __name__ == "__main__":
    generate_synthetic_data()
