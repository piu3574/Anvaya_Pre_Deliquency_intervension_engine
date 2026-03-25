import { useEffect, useState, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell
} from 'recharts'
import StatCard from '../components/StatCard'
import { fetchCustomers, scoreCustomer } from '../api/client'
import type { ScoreResult } from '../api/client'

// Colour lookup for score bars
function barColour(pd: number) {
  if (pd < 0.15) return '#3B82F6'
  if (pd < 0.35) return '#60A5FA'
  return '#1E40AF'
}

export default function Dashboard() {
  const [scores, setScores] = useState<ScoreResult[]>([])
  const [customerCount, setCustomerCount] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true
    ;(async () => {
      try {
        const res = await fetchCustomers()
        const customers = res.data
        if (!mounted) return
        setCustomerCount(customers.length)

        // Score first 20 to populate the dashboard stats
        const sample = customers.slice(0, 20)
        const scored: ScoreResult[] = []
        for (const c of sample) {
          try {
            const s = await scoreCustomer(c.customer_id)
            scored.push(s.data)
          } catch { /* skip */ }
        }
        if (mounted) { setScores(scored); setLoading(false) }
      } catch {
        if (mounted) setLoading(false)
      }
    })()
    return () => { mounted = false }
  }, [])

  const stats = useMemo(() => {
    const red    = scores.filter(s => s.band === 'RED').length
    const yellow = scores.filter(s => s.band === 'YELLOW').length
    const green  = scores.filter(s => s.band === 'GREEN').length
    return { red, yellow, green }
  }, [scores])

  // Build histogram bins 0.00..0.95 step 0.05
  const histData = useMemo(() => {
    const bins = Array.from({ length: 20 }, (_, i) => ({
      label: (i * 0.05).toFixed(2),
      count: 0,
    }))
    scores.forEach(s => {
      const idx = Math.min(Math.floor(s.pd_final / 0.05), 19)
      bins[idx].count++
    })
    return bins
  }, [scores])

  const highRisk = scores.filter(s => s.band === 'RED').slice(0, 5)

  return (
    <>
      {/* Page Header */}
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <div className="page-eyebrow">Anvaya Intelligence Cockpit</div>
            <div className="page-title">Pre-Delinquency Command Center</div>
            <div className="page-subtitle">
              Monitor customer risk signals, triage emerging alerts, and track pipeline confidence.
            </div>
            <div className="page-tags">
              <span className="page-tag"><span className="page-tag-dot" />Ensemble XGB + LGBM</span>
              <span className="page-tag"><span className="page-tag-dot" />SHAP Explainability</span>
              <span className="page-tag"><span className="page-tag-dot" />{customerCount} customers loaded</span>
            </div>
          </div>
        </div>
      </div>

      {/* Stat Cards */}
      <div className="stat-grid">
        <StatCard
          label="Customers Analysed"
          value={loading ? '—' : customerCount}
          meta="supabase live fetch"
          sub="scored in last pipeline run"
          color="blue"
          metaVariant="positive"
          icon={<svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}><path strokeLinecap="round" strokeLinejoin="round" d="M17 20h5v-1a4 4 0 00-5-3.87M9 20H4v-1a4 4 0 015-3.87M15 7a4 4 0 11-8 0 4 4 0 018 0z"/></svg>}
        />
        <StatCard
          label="High-Risk Alerts"
          value={loading ? '—' : stats.red}
          meta="above 35% PD threshold"
          sub="requires triage attention"
          color="red"
          metaVariant="danger"
          icon={<svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>}
        />
        <StatCard
          label="Watch List"
          value={loading ? '—' : stats.yellow}
          meta="15–35% PD — monitor"
          sub="RM notification queued"
          color="yellow"
          metaVariant="warning"
          icon={<svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path strokeLinecap="round" strokeLinejoin="round" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>}
        />
        <StatCard
          label="Safe Customers"
          value={loading ? '—' : stats.green}
          meta="below 15% PD"
          sub="exited via green path"
          color="green"
          metaVariant="success"
          icon={<svg fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7"/></svg>}
        />
      </div>

      {/* Charts Row */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr', gap: '20px' }}>

        {/* Risk Score Distribution */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Risk Score Distribution</div>
              <div className="card-subtitle">PD spread across low, medium, and high-risk cohorts</div>
            </div>
          </div>
          <div className="card-body">
            {loading ? (
              <div className="loading-spinner"><div className="spinner-ring" /><span>Scoring customers…</span></div>
            ) : scores.length === 0 ? (
              <div className="empty-state">No scored customers yet — run the pipeline.</div>
            ) : (
              <div className="chart-container" style={{ height: 220 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#EFF6FF" />
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#94A3B8' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#94A3B8' }} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{ border: 'none', borderRadius: 8, boxShadow: '0 4px 12px rgba(0,0,0,0.12)', fontSize: 12 }}
                      labelFormatter={(v) => `PD: ${v}`}
                    />
                    <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                      {histData.map((entry, index) => (
                        <Cell key={index} fill={barColour(parseFloat(entry.label))} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>

        {/* High-Risk Queue */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">High-Risk Queue</div>
              <div className="card-subtitle">Most recent detections above threshold 0.35</div>
            </div>
            <a href="/customers" className="card-link">
              View all →
            </a>
          </div>
          <div className="card-body" style={{ paddingTop: 8 }}>
            {loading ? (
              <div className="loading-spinner"><div className="spinner-ring" /></div>
            ) : highRisk.length === 0 ? (
              <div className="empty-state">No high-risk customers detected.</div>
            ) : (
              highRisk.map((s) => (
                <div className="risk-queue-item" key={s.customer_id}>
                  <div style={{ flex: 1 }}>
                    <div className="rq-id">{s.customer_id}</div>
                    <div className="rq-sub">{s.top_drivers[0]?.reason_code ?? 'Unassigned'}</div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span className="rq-score">{s.pd_final.toFixed(2)}</span>
                    <span className="rq-new-badge">NEW</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </>
  )
}
