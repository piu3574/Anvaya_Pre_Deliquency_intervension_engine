"""
generator.py — Phase-7 Raw Data Feature Generation Layer
Derives F1-F14 indicators from raw operational tables (transactions, loans, etc.).
"""
import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional

class FeatureGenerator:
    def __init__(self, db_client):
        self.db = db_client

    def get_feature_vector(self, customer_id: str, snapshot_date: datetime.date) -> Dict[str, float]:
        """
        Calculates all 13 features (F1-F14) for a given customer and date.
        """
        # 1. Fetch relevant windows of raw data
        # Window sizes (in days)
        W_3M = 90
        W_6M = 180
        
        # 2. Derive Features
        features = {}
        
        # F1 - EMI to Income Ratio
        # Logic: avg(emi_due) / avg(income) over 3m
        features['f1'] = self._compute_f1(customer_id, snapshot_date)
        
        # F2 - Savings Drawdown
        # Logic: (bal_T-6m - bal_T) / bal_T-6m
        features['f2'] = self._compute_f2(customer_id, snapshot_date)
        
        # F3 - Salary Delay Days
        features['f3'] = self._compute_f3(customer_id, snapshot_date)
        
        # F4 - Spending Pattern Shift
        features['f4'] = self._compute_f4(customer_id, snapshot_date)
        
        # F5 - Auto-Debit Failure Count
        features['f5'] = self._compute_f5(customer_id, snapshot_date)
        
        # F6 - Lending App Transaction Count
        features['f6'] = self._compute_f6(customer_id, snapshot_date)
        
        # F7 - Cash Hoarding Ratio
        features['f7'] = self._compute_f7(customer_id, snapshot_date)
        
        # F8 - Stress Velocity
        features['f8'] = self._compute_f8(customer_id, snapshot_date)
        
        # F9 - Payment Timing Entropy
        features['f9'] = self._compute_f9(customer_id, snapshot_date)
        
        # F10 - Peer Cohort Stress Index
        features['f10'] = self._compute_f10(customer_id, snapshot_date)
        
        # F12 - Cross Loan Consistency
        features['f12'] = self._compute_f12(customer_id, snapshot_date)
        
        # F13 - Secondary Income Index
        features['f13'] = self._compute_f13(customer_id, snapshot_date)
        
        # F14 - Revolving Credit Utilisation
        features['f14'] = self._compute_f14(customer_id, snapshot_date)
        
        return features

    # --- Feature Derivation Methods ---

    def _compute_f1(self, cid, T):
        """F1 - EMI to Income Ratio (Avg 3 months)"""
        try:
            # Query loans for EMI sum
            loans_res = self.db.table("loans_raw")\
                .select("emi_amount")\
                .eq("customer_id", cid)\
                .gte("due_date", (T - datetime.timedelta(days=90)).isoformat())\
                .execute()
            
            # Query salary for income sum
            sal_res = self.db.table("salary_raw")\
                .select("amount")\
                .eq("customer_id", cid)\
                .gte("credit_date", (T - datetime.timedelta(days=90)).isoformat())\
                .execute()
            
            avg_emi = sum([r['emi_amount'] for r in loans_res.data]) / 3
            avg_inc = sum([r['amount'] for r in sal_res.data]) / 3
            
            if avg_inc == 0: return 2.0 # Cap for high risk
            return min(2.0, avg_emi / avg_inc)
        except Exception as e:
            print(f"Error computing F1: {e}")
            return 0.0

    def _compute_f5(self, cid, T):
        """F5 - Auto-Debit Failure Count (3 months)"""
        try:
            res = self.db.table("loans_raw")\
                .select("failed_attempts")\
                .eq("customer_id", cid)\
                .eq("status", "failed")\
                .gte("due_date", (T - datetime.timedelta(days=90)).isoformat())\
                .execute()
            return float(len(res.data))
        except Exception as e:
            return 0.0

    def _compute_f2(self, cid, T):
        """F2 - Savings Drawdown (6 months window)"""
        try:
            # (bal_start - bal_end) / bal_start
            t_start = (T - datetime.timedelta(days=180)).isoformat()
            
            res_start = self.db.table("savings_raw").select("balance").eq("customer_id", cid).gte("snapshot_date", t_start).order("snapshot_date").limit(1).execute()
            res_end = self.db.table("savings_raw").select("balance").eq("customer_id", cid).lte("snapshot_date", T.isoformat()).order("snapshot_date", desc=True).limit(1).execute()
            
            if not res_start.data or not res_end.data: return 0.0
            bs = float(res_start.data[0]['balance'])
            be = float(res_end.data[0]['balance'])
            
            if bs == 0: return 1.0 # Max drawdown if start was 0 but end is 0
            return (bs - be) / bs
        except Exception: return 0.0

    def _compute_f3(self, cid, T):
        """F3 - Salary Delay Days (Avg over last 3 cycles)"""
        try:
            res = self.db.table("salary_raw").select("credit_date").eq("customer_id", cid).gte("credit_date", (T - datetime.timedelta(days=90)).isoformat()).execute()
            if len(res.data) < 2: return 0.0
            # Simplified: diff between actual dates in days
            dates = sorted([datetime.datetime.strptime(r['credit_date'], "%Y-%m-%d") for r in res.data])
            diffs = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
            # Delay defined as days beyond 30-day cycle
            delays = [max(0, d - 30) for d in diffs]
            return sum(delays) / len(delays) if delays else 0.0
        except Exception: return 0.0

    def _compute_f4(self, cid, T):
        """F4 - Spending Pattern Shift (Distance between 3m and 6m distributions)"""
        try:
            t3 = (T - datetime.timedelta(days=90)).isoformat()
            t6 = (T - datetime.timedelta(days=180)).isoformat()
            
            # Recent window (0-3m)
            res_rec = self.db.table("transactions_raw").select("amount, category").eq("customer_id", cid).gte("txn_date", t3).execute()
            # Older window (3-6m)
            res_old = self.db.table("transactions_raw").select("amount, category").eq("customer_id", cid).gte("txn_date", t6).lt("txn_date", t3).execute()
            
            if not res_rec.data or not res_old.data: return 0.0
            
            def get_dist(data):
                df = pd.DataFrame(data)
                return df.groupby('category')['amount'].sum() / df['amount'].sum()
            
            d_rec = get_dist(res_rec.data)
            d_old = get_dist(res_old.data)
            
            # Euclidean distance between distributions
            all_cats = list(set(d_rec.index) | set(d_old.index))
            v_rec = np.array([d_rec.get(c, 0.0) for c in all_cats])
            v_old = np.array([d_old.get(c, 0.0) for c in all_cats])
            
            return np.linalg.norm(v_rec - v_old)
        except Exception: return 0.0

    def _compute_f6(self, cid, T):
        """F6 - Lending App Transaction Count (6 months)"""
        try:
            res = self.db.table("transactions_raw").select("id").eq("customer_id", cid).eq("category", "lending_app").gte("txn_date", (T - datetime.timedelta(days=180)).isoformat()).execute()
            return float(len(res.data))
        except Exception: return 0.0

    def _compute_f7(self, cid, T):
        """F7 - Cash Hoarding Ratio (3 months)"""
        try:
            res = self.db.table("transactions_raw").select("amount, category").eq("customer_id", cid).gte("txn_date", (T - datetime.timedelta(days=90)).isoformat()).execute()
            if not res.data: return 0.0
            cash = sum([r['amount'] for r in res.data if r['category'] == 'cash'])
            total = sum([r['amount'] for r in res.data])
            return cash / total if total > 0 else 0.0
        except Exception: return 0.0

    def _compute_f8(self, cid, T):
        """F8 - Stress Velocity (Slope of balance over 3 months)"""
        try:
            res = self.db.table("balances_raw").select("snapshot_date, current_balance").eq("customer_id", cid).gte("snapshot_date", (T - datetime.timedelta(days=90)).isoformat()).execute()
            if len(res.data) < 2: return 0.0
            df = pd.DataFrame(res.data)
            df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
            df = df.sort_values('snapshot_date')
            # Simply: (End Bal - Start Bal) / Start Bal (normalized decay)
            start_bal = float(df.iloc[0]['current_balance'])
            end_bal = float(df.iloc[-1]['current_balance'])
            if start_bal == 0: return 0.0
            return (start_bal - end_bal) / start_bal # Positive means balance is dropping (stress)
        except Exception: return 0.0

    def _compute_f9(self, cid, T):
        """F9 - Payment Timing Entropy (Std Dev of payment days)"""
        try:
            res = self.db.table("loans_raw").select("paid_date").eq("customer_id", cid).eq("status", "paid").gte("paid_date", (T - datetime.timedelta(days=180)).isoformat()).execute()
            if len(res.data) < 3: return 0.0
            days = [datetime.datetime.strptime(r['paid_date'], "%Y-%m-%d").day for r in res.data]
            return float(np.std(days))
        except Exception: return 0.0

    def _compute_f10(self, cid, T):
        """F10 - Peer Cohort Stress Index (Based on employment category)"""
        try:
            # First find customer's cohort
            cust_res = self.db.table("accounts_raw").select("employment_category").eq("customer_id", cid).execute()
            if not cust_res.data: return 0.0
            cohort = cust_res.data[0]['employment_category']
            
            # Find all customers in this cohort
            peers_res = self.db.table("accounts_raw").select("customer_id").eq("employment_category", cohort).execute()
            peer_ids = [r['customer_id'] for r in peers_res.data]
            
            # Count defaults/missed payments in cohort over 6m
            bad_res = self.db.table("loans_raw")\
                .select("customer_id")\
                .in_("customer_id", peer_ids)\
                .eq("status", "missed")\
                .gte("due_date", (T - datetime.timedelta(days=180)).isoformat())\
                .execute()
            
            if not peer_ids: return 0.0
            return len(set([r['customer_id'] for r in bad_res.data])) / len(peer_ids)
        except Exception: return 0.0

    def _compute_f12(self, cid, T):
        """F12 - Cross Loan Consistency (Scheduled vs Paid)"""
        try:
            res = self.db.table("loans_raw").select("status").eq("customer_id", cid).gte("due_date", (T - datetime.timedelta(days=180)).isoformat()).execute()
            if not res.data: return 1.0
            paid = len([r for r in res.data if r['status'] == 'paid'])
            return paid / len(res.data)
        except Exception: return 1.0

    def _compute_f13(self, cid, T):
        """F13 - Secondary Income Index (Non-Salary credits ratio)"""
        try:
            res = self.db.table("transactions_raw").select("amount, category").eq("customer_id", cid).eq("txn_type", "credit").gte("txn_date", (T - datetime.timedelta(days=90)).isoformat()).execute()
            if not res.data: return 0.0
            total_credit = sum([r['amount'] for r in res.data])
            salary_credit = sum([r['amount'] for r in res.data if r['category'] == 'salary'])
            if salary_credit == 0: return 0.0
            return (total_credit - salary_credit) / salary_credit
        except Exception: return 0.0

    def _compute_f14(self, cid, T):
        """F14 - Revolving Credit Utilisation (Current / Limit)"""
        try:
            res = self.db.table("balances_raw").select("current_balance, credit_limit").eq("customer_id", cid).order("snapshot_date", desc=True).limit(1).execute()
            if not res.data: return 0.0
            cb = float(res.data[0]['current_balance'])
            cl = float(res.data[0]['credit_limit'])
            if cl == 0: return 1.5 # Capped
            return min(1.5, abs(cb) / cl if cb < 0 else 0.0) # Assume negative balance is credit card debt
        except Exception: return 0.0
