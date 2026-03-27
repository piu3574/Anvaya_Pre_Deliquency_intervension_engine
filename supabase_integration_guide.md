# Anvaya Supabase Integration Guide

This document contains everything you need to execute in your Supabase SQL editor and the Python script needed to push your 30,000 local test rows directly into your production database.

---

## 1. Supabase SQL DDL (Table Schemas)
*Copy and paste this absolutely intact directly into your Supabase SQL Editor and hit "Run".*

```sql
-- TABLE 1: RAW INPUTS
CREATE TABLE IF NOT EXISTS public.anvaya_test_raw (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT UNIQUE NOT NULL,
    age INTEGER,
    gender TEXT,
    customer_type TEXT,
    region TEXT,
    employment_sector TEXT,
    
    primary_income_monthly NUMERIC(18,2),
    has_secondary_income BOOLEAN,
    secondary_income_typical_monthly NUMERIC(18,2),
    has_credit_card BOOLEAN,
    credit_card_limit_total NUMERIC(18,2),
    has_overdraft_facility BOOLEAN,
    overdraft_limit NUMERIC(18,2),
    
    num_active_loans INTEGER,
    total_loan_principal NUMERIC(18,2),
    total_monthly_emi_due NUMERIC(18,2),
    max_single_emi_amount NUMERIC(18,2),
    
    avg_monthly_salary_last_3m NUMERIC(18,2),
    avg_monthly_income_last_3m NUMERIC(18,2),
    balance_60d_ago NUMERIC(18,2),
    current_balance NUMERIC(18,2),
    num_salary_credits_last_3m INTEGER,
    expected_salary_day INTEGER,
    avg_actual_salary_day_last_3m INTEGER,
    
    total_atm_withdrawal_last_30d NUMERIC(18,2),
    total_spend_last_30d NUMERIC(18,2),
    num_auto_debits_last_30d INTEGER,
    num_auto_debit_failures_last_30d INTEGER,
    num_lending_app_txn_last_30d INTEGER,
    num_overdraft_days_last_30d INTEGER,
    
    num_emi_payments_last_6m INTEGER,
    stddev_emi_payment_delay_days_last_6m NUMERIC(18,4),
    
    revolving_credit_limit_total NUMERIC(18,2),
    revolving_credit_balance_current NUMERIC(18,2),
    
    sector_stress_index NUMERIC(6,4),
    region_stress_index NUMERIC(6,4),
    
    default_label SMALLINT
);

-- INDEXES for fast grouping
CREATE INDEX idx_raw_customer_type ON public.anvaya_test_raw(customer_type);
CREATE INDEX idx_raw_default_label ON public.anvaya_test_raw(default_label);


-- TABLE 2: MODEL RESULTS & SHAP
CREATE TABLE IF NOT EXISTS public.anvaya_test_results (
    id BIGSERIAL PRIMARY KEY,
    customer_id TEXT UNIQUE REFERENCES public.anvaya_test_raw(customer_id) ON DELETE CASCADE,
    
    model_version TEXT,
    xgb_model_hash TEXT,
    lgbm_model_hash TEXT,
    ensemble_model_hash TEXT,
    scored_at TIMESTAMPTZ DEFAULT NOW(),
    
    pd_xgboost NUMERIC(6,4),
    pd_lightgbm NUMERIC(6,4),
    pd_final NUMERIC(6,4),
    
    risk_band TEXT CHECK (risk_band IN ('GREEN', 'YELLOW', 'RED')),
    predicted_default_flag BOOLEAN,
    true_default_label SMALLINT,
    
    confusion_bucket TEXT CHECK (confusion_bucket IN ('TP', 'FP', 'TN', 'FN')),
    
    shap_top_features JSONB,
    top_positive_driver TEXT,  -- The feature driving probability UP the most
    top_negative_driver TEXT   -- The feature pulling probability DOWN the most
);

-- INDEXES for dashboard filtering
CREATE INDEX idx_results_risk_band ON public.anvaya_test_results(risk_band);
CREATE INDEX idx_results_confusion ON public.anvaya_test_results(confusion_bucket);
CREATE INDEX idx_results_pd_final ON public.anvaya_test_results(pd_final);


-- 3. ROW LEVEL SECURITY (RLS) POLICIES
ALTER TABLE public.anvaya_test_raw ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.anvaya_test_results ENABLE ROW LEVEL SECURITY;

-- Allow python ingestion keys (authenticated) to execute everything
CREATE POLICY "Allow authenticated full access raw" ON public.anvaya_test_raw FOR ALL TO authenticated USING (true);
CREATE POLICY "Allow authenticated full access res" ON public.anvaya_test_results FOR ALL TO authenticated USING (true);

-- Allow frontend anonymous dashboard queries to ONLY SELECT tables safely
CREATE POLICY "Allow anon select raw" ON public.anvaya_test_raw FOR SELECT TO anon USING (true);
CREATE POLICY "Allow anon select res" ON public.anvaya_test_results FOR SELECT TO anon USING (true);
```

---

## 2. Python Bulk Upload Script (Batch Processing)
*Run `pip install supabase pandas numpy` if needed. Ensure your dataframe maps closely to the columns above.*

```python
import os
import pandas as pd
import numpy as np
from supabase import create_client, Client

SUPABASE_URL = "https://ckcxagnpxypowuswptir.supabase.co"
SUPABASE_KEY = "eyJhbGci... <PASTE YOUR FULL KEY HERE>"  

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 1. LOAD YOUR 30k DF
df_raw = pd.read_csv("my_30k_test_raw.csv")
df_results = pd.read_csv("my_30k_test_results.csv")

# Ensure NaN values are replaced with None (valid for Supabase JSON/SQL nulls)
df_raw = df_raw.replace({np.nan: None})
df_results = df_results.replace({np.nan: None})

def push_table_in_batches(df, table_name, batch_size=1000):
    records = df.to_dict(orient="records")
    total = len(records)
    print(f"Uploading {total} records to {table_name}...")
    
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        try:
            # Using upsert ensures if you re-run it, it overwrites duplicates instantly
            response = supabase.table(table_name).upsert(batch).execute()
            print(f"Uploaded batch {i} -> {i+len(batch)} safely.")
        except Exception as e:
            print(f"Failed at batch {i}: {e}")

# IMPORTANT: Push RAW first (since Results Foreign Key depends on it)
push_table_in_batches(df_raw, 'anvaya_test_raw')
push_table_in_batches(df_results, 'anvaya_test_results')
print("✅ Supabase Synchronization Complete.")
```

---

## 3. Example High-Value SQL Analytics Queries

#### Query 1: Band-Wise Default Rates 
*Perfect for standard Dashboard Pie Charts & Tables.*
```sql
SELECT 
    risk_band,
    COUNT(*) as total_customers,
    SUM(true_default_label) as total_defaulters,
    ROUND((SUM(true_default_label)::numeric / COUNT(*)) * 100, 2) as default_rate_pct
FROM 
    anvaya_test_results
GROUP BY 
    risk_band
ORDER BY 
    default_rate_pct ASC;
```

#### Query 2: Confusion Matrix Isolation
*Finds exactly where your model lost accuracy (Missed Defaulters).*
```sql
SELECT 
    confusion_bucket,
    COUNT(*) as volume
FROM 
    anvaya_test_results
GROUP BY 
    confusion_bucket;
```

#### Query 3: Show 5 Sample `RED` Customers Joined With their Behaviors
*Finds 5 extreme high-risk customers, pulling exactly what caused their score.*
```sql
SELECT 
    r.customer_id,
    res.pd_final,
    res.top_positive_driver,
    r.num_auto_debit_failures_last_30d,
    r.revolving_credit_balance_current,
    r.total_monthly_emi_due
FROM 
    anvaya_test_results res
JOIN 
    anvaya_test_raw r ON res.customer_id = r.customer_id
WHERE 
    res.risk_band = 'RED'
ORDER BY 
    res.pd_final DESC
LIMIT 5;
```
