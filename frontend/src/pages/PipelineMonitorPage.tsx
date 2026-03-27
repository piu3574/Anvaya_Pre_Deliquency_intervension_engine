import { useEffect, useState } from "react";
import { getPipelineRuns, getPipelineStagesLatest, getPipelineStatus, getRiskLogs } from "../api/client";
import type { PipelineRun, PipelineStage, PipelineStatus, RiskLogItem } from "../types";

export default function PipelineMonitorPage() {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [stages, setStages] = useState<PipelineStage[]>([]);
  const [logs, setLogs] = useState<RiskLogItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setError(null);
      try {
        const [statusData, runsData, stagesData, logsData] = await Promise.all([
          getPipelineStatus(),
          getPipelineRuns(24),
          getPipelineStagesLatest(),
          getRiskLogs(20, 0)
        ]);
        setStatus(statusData);
        setRuns(runsData);
        setStages(stagesData);
        setLogs(logsData.items);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load pipeline monitor");
      }
    }
    load();
  }, []);

  return (
    <div className="dashboard-view">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Execution Intelligence</p>
          <h1>Pipeline Monitor</h1>
          <p className="hero-subtitle">Monitor end-to-end scoring flow, stage health, and risk-event throughput in real time.</p>
          <div className="chip-row">
            <span className="chip">State: {status?.state ?? "unknown"}</span>
            <span className="chip">Events (15m): {status?.events_last_15m ?? 0}</span>
            <span className="chip">RED events (15m): {status?.red_events_last_15m ?? 0}</span>
          </div>
        </div>
      </section>

      {error ? <div className="alert-banner">{error}</div> : null}

      <section className="dashboard-bottom-grid">
        <article className="panel">
          <h2>Stage Status</h2>
          <p>Current execution state across core scoring stages</p>
          <div className="simple-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                {stages.map((stage) => (
                  <tr key={stage.stage}>
                    <td>{stage.stage}</td>
                    <td>{stage.name}</td>
                    <td>
                      <span className={`tag ${stage.status === "ok" ? "tag-green" : "tag-red"}`}>{stage.status}</span>
                    </td>
                    <td>{stage.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="panel">
          <h2>Recent Run Windows</h2>
          <p>Hourly execution windows with critical event density</p>
          <div className="simple-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Window</th>
                  <th>Total</th>
                  <th>Critical</th>
                  <th>Watch</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {runs.slice(0, 12).map((run) => (
                  <tr key={run.run_window}>
                    <td>{new Date(run.run_window).toLocaleString()}</td>
                    <td>{run.total_events}</td>
                    <td>{run.critical_events}</td>
                    <td>{run.watch_events}</td>
                    <td>
                      <span className={`tag ${run.status === "healthy" ? "tag-green" : "tag-red"}`}>{run.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>

      <section className="panel">
        <h2>Latest Scoring Logs</h2>
        <p>Most recent model-scored customers from risk log stream source</p>
        <div className="simple-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Customer</th>
                <th>PD</th>
                <th>Risk</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((log) => (
                <tr key={`${log.id}-${log.scored_at}`}>
                  <td>{new Date(log.scored_at).toLocaleString()}</td>
                  <td>{log.customer_id}</td>
                  <td>{log.pd_ensemble.toFixed(4)}</td>
                  <td>
                    <span
                      className={`tag ${
                        log.risk_band === "RED" ? "tag-red" : log.risk_band === "YELLOW" ? "tag-amber" : "tag-green"
                      }`}
                    >
                      {log.risk_band}
                    </span>
                  </td>
                  <td>{log.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
