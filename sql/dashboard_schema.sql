-- Dashboard Foundation: Source of Truth Table
-- Designed for 30,000 baseline test-set records

CREATE TABLE IF NOT EXISTS dashboard_customers (
    customer_id TEXT PRIMARY KEY,
    pd_final FLOAT8 NOT NULL,
    risk_band TEXT NOT NULL,
    y_true INT4 NOT NULL,
    pd_xgb FLOAT8,
    pd_lgbm FLOAT8,
    
    -- Feature Indicators (snake_case)
    f1_emi_to_income FLOAT8,
    f2_savings_drawdown FLOAT8,
    f3_salary_delay FLOAT8,
    f4_spend_shift FLOAT8,
    f5_auto_debit_fails FLOAT8,
    f6_lending_app_usage FLOAT8,
    f7_overdraft_freq FLOAT8,
    f8_stress_velocity FLOAT8,
    f9_payment_entropy FLOAT8,
    f14_active_loan_pressure FLOAT8,
    f10_peer_stress FLOAT8,
    f12_cross_loan FLOAT8,
    f13_secondary_income FLOAT8,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Performance Indices for Dashboard Filtering
CREATE INDEX IF NOT EXISTS idx_dashboard_risk_band ON dashboard_customers(risk_band);
CREATE INDEX IF NOT EXISTS idx_dashboard_pd_final ON dashboard_customers(pd_final);
