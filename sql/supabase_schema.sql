-- 1. Create Tables

-- Customers Table
CREATE TABLE IF NOT EXISTS customers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Risk Scores Table (Logs every prediction)
CREATE TABLE IF NOT EXISTS risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID NULL REFERENCES customers(id),
    external_customer_id TEXT NOT NULL,
    pd_xgb NUMERIC NOT NULL,
    pd_lgbm NUMERIC NOT NULL,
    pd_ensemble NUMERIC NOT NULL,
    risk_band TEXT NOT NULL,
    t_green NUMERIC NOT NULL,
    t_red NUMERIC NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source TEXT DEFAULT 'api_v1_ensemble'
);

-- Risk Explanations Table (Logs top SHAP drivers per score)
CREATE TABLE IF NOT EXISTS risk_explanations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_score_id UUID NOT NULL REFERENCES risk_scores(id) ON DELETE CASCADE,
    feature_name TEXT NOT NULL,
    shap_value NUMERIC NOT NULL,
    rank INT NOT NULL
);

-- 2. Security (RLS)
ALTER TABLE risk_scores ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_explanations ENABLE ROW LEVEL SECURITY;

-- 3. Example RLS Policies (Read-only for Authenticated Users/Dashboards)

CREATE POLICY "Allow authenticated users to select risk_scores" 
ON risk_scores FOR SELECT 
TO authenticated 
USING (true);

CREATE POLICY "Allow authenticated users to select risk_explanations" 
ON risk_explanations FOR SELECT 
TO authenticated 
USING (true);

-- Note: Backend uses SERVICE_ROLE_KEY which bypasses RLS for INSERTs.
