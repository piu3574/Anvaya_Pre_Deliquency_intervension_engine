import { useEffect, useState } from 'react'
import RiskBadge from '../components/RiskBadge'
import { fetchCustomers, scoreCustomer } from '../api/client'
import type { Customer, ScoreResult } from '../api/client'

export default function CustomerList() {
  const [customers, setCustomers] = useState<Customer[]>([])
  const [scores, setScores] = useState<Record<string, ScoreResult>>({})
  const [loading, setLoading] = useState(true)
  const [scoring, setScoring] = useState(false)
  const [search, setSearch] = useState('')

  useEffect(() => {
    fetchCustomers()
      .then(r => { setCustomers(r.data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const runScoreAll = async () => {
    setScoring(true)
    for (const c of customers) {
      try {
        const r = await scoreCustomer(c.customer_id)
        setScores(prev => ({ ...prev, [c.customer_id]: r.data }))
      } catch { /* skip */ }
    }
    setScoring(false)
  }

  const filtered = customers.filter(c =>
    c.customer_id.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <div className="page-eyebrow">Anvaya</div>
            <div className="page-title">Customer Risk List</div>
            <div className="page-subtitle">
              All customers fetched from Supabase with real-time PD scores.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <div className="search-bar">
              <svg className="search-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
              </svg>
              <input
                placeholder="Search customer ID…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <button
              className="btn btn-primary"
              onClick={runScoreAll}
              disabled={scoring || loading}
            >
              {scoring ? (
                <>
                  <div className="spinner-ring" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  Scoring…
                </>
              ) : (
                <>
                  <svg style={{ width: 14, height: 14 }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                  Score All
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="table-wrap">
          {loading ? (
            <div className="loading-spinner"><div className="spinner-ring" /><span>Loading customers…</span></div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th>Customer ID</th>
                  <th>Risk Band</th>
                  <th>PD Final</th>
                  <th>PD XGB</th>
                  <th>PD LGBM</th>
                  <th>Top Driver</th>
                  <th>Scored At</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(c => {
                  const s = scores[c.customer_id]
                  return (
                    <tr key={c.customer_id}>
                      <td><span className="customer-id-cell">{c.customer_id}</span></td>
                      <td>{s ? <RiskBadge band={s.band} /> : <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>—</span>}</td>
                      <td>
                        {s ? (
                          <div className="pd-gauge-wrap">
                            <div className="pd-bar-outer">
                              <div
                                className={`pd-bar-inner ${s.band}`}
                                style={{ width: `${Math.round(s.pd_final * 100)}%` }}
                              />
                            </div>
                            <span className="score-cell">{(s.pd_final * 100).toFixed(1)}%</span>
                          </div>
                        ) : '—'}
                      </td>
                      <td className="score-cell">{s ? `${(s.pd_xgb * 100).toFixed(1)}%` : '—'}</td>
                      <td className="score-cell">{s ? `${(s.pd_lgbm * 100).toFixed(1)}%` : '—'}</td>
                      <td style={{ fontSize: 11, color: 'var(--text-secondary)', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {s ? s.top_drivers[0]?.reason_code ?? '—' : '—'}
                      </td>
                      <td style={{ color: 'var(--text-muted)', fontSize: 11 }}>
                        {s ? new Date(s.timestamp).toLocaleTimeString() : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  )
}
