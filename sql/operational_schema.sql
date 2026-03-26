-- Phase-7: Operational Raw Fact Tables
-- These tables store the raw "events" before they are aggregated into F1-F14 features.

-- 1. Accounts Master
CREATE TABLE accounts_raw (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id text UNIQUE NOT NULL,
    segment text,
    monthly_net_salary numeric,
    employment_category text,
    customer_vintage_months int,
    total_credit_limit numeric,
    created_at timestamptz DEFAULT now()
);

-- 2. Transactions (Debits/Credits)
CREATE TABLE transactions_raw (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id text REFERENCES accounts_raw(customer_id),
    txn_date date NOT NULL,
    amount numeric NOT NULL,
    txn_type text CHECK (txn_type IN ('debit', 'credit')),
    category text, -- 'food', 'travel', 'emi', 'cash', 'lending_app', etc.
    merchant_name text,
    created_at timestamptz DEFAULT now()
);

-- 3. Loans & EMI Schedules
CREATE TABLE loans_raw (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id text REFERENCES accounts_raw(customer_id),
    loan_id text NOT NULL,
    emi_amount numeric NOT NULL,
    due_date date NOT NULL,
    paid_date date,
    status text DEFAULT 'scheduled', -- 'paid', 'missed', 'failed'
    failed_attempts int DEFAULT 0,
    created_at timestamptz DEFAULT now()
);

-- 4. Balances & Limits
CREATE TABLE balances_raw (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id text REFERENCES accounts_raw(customer_id),
    snapshot_date date NOT NULL,
    current_balance numeric NOT NULL,
    credit_limit numeric,
    created_at timestamptz DEFAULT now()
);

-- 5. Salary Credits (Derived or specialized view)
CREATE TABLE salary_raw (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id text REFERENCES accounts_raw(customer_id),
    credit_date date NOT NULL,
    amount numeric NOT NULL,
    created_at timestamptz DEFAULT now()
);

-- 6. Savings Account History
CREATE TABLE savings_raw (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id text REFERENCES accounts_raw(customer_id),
    snapshot_date date NOT NULL,
    balance numeric NOT NULL,
    created_at timestamptz DEFAULT now()
);

-- Indexes for performance
CREATE INDEX idx_txn_cust_date ON transactions_raw(customer_id, txn_date);
CREATE INDEX idx_loans_cust_date ON loans_raw(customer_id, due_date);
CREATE INDEX idx_balances_cust_date ON balances_raw(customer_id, snapshot_date);
