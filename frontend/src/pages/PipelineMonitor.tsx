import { useState, useCallback } from 'react'
import LogConsole from '../components/LogConsole'
import type { LogEntry } from '../components/LogConsole'
import RiskBadge from '../components/RiskBadge'
import type { ScoreResult } from '../api/client'

function nowStr() {
  return new Date().toLocaleTimeString([], { hour12: false })
}

export default function PipelineMonitor() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [scores, setScores] = useState<ScoreResult[]>([])
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)
  const [es, setEs] = useState<EventSource | null>(null)

  const pushLog = useCallback((type: LogEntry['type'], body: string) => {
    setLogs(prev => [...prev, { ts: nowStr(), type, body }])
  }, [])

  const runPipeline = () => {
    if (running) return
    setLogs([])
    setScores([])
    setDone(false)
    setRunning(true)

    pushLog('info', 'Connecting to Anvaya pipeline stream…')

    const source = new EventSource('/api/pipeline/run')
    setEs(source)

    source.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)

        if (msg.type === 'log') {
          pushLog('info', msg.message)
        } else if (msg.type === 'score') {
          const d = msg.data as ScoreResult
          pushLog('score', `✓ ${d.customer_id} → PD: ${(d.pd_final * 100).toFixed(1)}% [${d.band}]`)
          setScores(prev => [...prev, d])
        } else if (msg.type === 'done') {
          pushLog('done', '✔ Pipeline complete. All customers scored.')
          source.close()
          setRunning(false)
          setDone(true)
        } else if (msg.type === 'error') {
          pushLog('error', `✗ Error: ${msg.message}`)
          source.close()
          setRunning(false)
        }
      } catch { /* skip */ }
    }

    source.onerror = () => {
      pushLog('error', '✗ Connection to pipeline stream lost.')
      source.close()
      setRunning(false)
    }
  }

  const stopPipeline = () => {
    es?.close()
    setEs(null)
    setRunning(false)
    pushLog('warn', '⚠ Pipeline manually stopped.')
  }

  const clearAll = () => {
    setLogs([])
    setScores([])
    setDone(false)
  }

  const redCount = scores.filter(s => s.band === 'RED').length
  const yellowCount = scores.filter(s => s.band === 'YELLOW').length
  const greenCount = scores.filter(s => s.band === 'GREEN').length

  return (
    <>
      <div className="page-header">
        <div className="page-header-row">
          <div>
            <div className="page-eyebrow">Anvaya</div>
            <div className="page-title">Pipeline Monitor</div>
            <div className="page-subtitle">
              Run the full ensemble scoring pipeline and watch live logs and risk scores.
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            {running && (
              <button className="btn btn-danger" onClick={stopPipeline}>
                <svg style={{ width: 14, height: 14 }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <rect x="6" y="6" width="12" height="12" rx="2"/>
                </svg>
                Stop
              </button>
            )}
            <button className="btn btn-outline" onClick={clearAll} disabled={running}>
              Clear
            </button>
            <button
              className="btn btn-primary"
              onClick={runPipeline}
              disabled={running}
            >
              {running ? (
                <>
                  <div className="spinner-ring" style={{ width: 14, height: 14, borderWidth: 2 }} />
                  Running…
                </>
              ) : (
                <>
                  <svg style={{ width: 14, height: 14 }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"/>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                  </svg>
                  ▶ Run Pipeline
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Summary row when done */}
      {done && scores.length > 0 && (
        <div className="stat-grid" style={{ marginBottom: 20 }}>
          <div className="stat-card">
            <div className="stat-card-label">Total Scored</div>
            <div className="stat-card-value">{scores.length}</div>
            <div className="stat-card-meta positive">pipeline complete</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">🔴 High Risk</div>
            <div className="stat-card-value" style={{ color: 'var(--red)' }}>{redCount}</div>
            <div className="stat-card-meta danger">PD &gt; 35%</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">🟡 Watch List</div>
            <div className="stat-card-value" style={{ color: 'var(--yellow)' }}>{yellowCount}</div>
            <div className="stat-card-meta warning">PD 15–35%</div>
          </div>
          <div className="stat-card">
            <div className="stat-card-label">🟢 Safe</div>
            <div className="stat-card-value" style={{ color: 'var(--green)' }}>{greenCount}</div>
            <div className="stat-card-meta success">PD &lt; 15%</div>
          </div>
        </div>
      )}

      <div className="pipeline-grid">
        {/* Log console */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Live Pipeline Logs</div>
              <div className="card-subtitle">Real-time streaming output from the scoring engine</div>
            </div>
            <span style={{ fontSize: 11, color: running ? 'var(--blue-500)' : 'var(--text-muted)', fontWeight: 600 }}>
              {running ? '● LIVE' : logs.length > 0 ? '◉ STOPPED' : '○ IDLE'}
            </span>
          </div>
          <div className="card-body">
            <LogConsole logs={logs} />
          </div>
        </div>

        {/* Scores table */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Scored Customers</div>
              <div className="card-subtitle">Results accumulate as the pipeline runs</div>
            </div>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{scores.length} scored</span>
          </div>
          <div className="table-wrap" style={{ maxHeight: 420, overflowY: 'auto' }}>
            {scores.length === 0 ? (
              <div className="empty-state">Results will appear here while the pipeline runs.</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Customer ID</th>
                    <th>Band</th>
                    <th>PD</th>
                    <th>Top Driver</th>
                  </tr>
                </thead>
                <tbody>
                  {scores.map((s, i) => (
                    <tr key={i}>
                      <td><span className="customer-id-cell">{s.customer_id}</span></td>
                      <td><RiskBadge band={s.band} /></td>
                      <td className="score-cell">{(s.pd_final * 100).toFixed(1)}%</td>
                      <td style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                        {s.top_drivers?.[0]?.reason_code ?? '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </>
  )
}
