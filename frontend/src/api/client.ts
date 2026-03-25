import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
})

export interface Customer {
  customer_id: string
  F1_emi_to_income?: number
  F2_savings_drawdown?: number
  F3_salary_delay?: number
  F4_spend_shift?: number
  F5_auto_debit_fails?: number
  F6_lending_app_usage?: number
  F7_overdraft_freq?: number
  F8_stress_velocity?: number
  F9_payment_entropy?: number
  F10_peer_stress?: number
  F12_cross_loan?: number
  F13_secondary_income?: number
  F14_active_loan_pressure?: number
}

export interface ScoreResult {
  customer_id: string
  pd_xgb: number
  pd_lgbm: number
  pd_final: number
  band: 'GREEN' | 'YELLOW' | 'RED'
  top_drivers: Array<{
    feature: string
    direction: 'up' | 'down'
    value: number
    reason_code: string
  }>
  timestamp: string
}

export const fetchCustomers = () => api.get<Customer[]>('/customers')
export const scoreCustomer = (id: string) => api.get<ScoreResult>(`/score/${id}`)
