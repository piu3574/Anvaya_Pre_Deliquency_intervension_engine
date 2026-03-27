import { useEffect, useMemo, useState } from "react";
import { getCustomers, getRiskLogs } from "../api/client";
import type { CustomerRow, RiskLogItem, RiskBand } from "../types";

interface PartyRiskListPageProps {
  logsOnly?: boolean;
}

export default function PartyRiskListPage({ logsOnly = false }: PartyRiskListPageProps) {
  const [band, setBand] = useState<"ALL" | RiskBand>("ALL");
  const [customers, setCustomers] = useState<CustomerRow[]>([]);
  const [logs, setLogs] = useState<RiskLogItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setError(null);
      try {
        const selectedBand = band === "ALL" ? undefined : band;
        const [customerRows, logRows] = await Promise.all([
          logsOnly ? Promise.resolve([]) : getCustomers(200, 0, selectedBand),
          getRiskLogs(150, 0, selectedBand)
        ]);
        setCustomers(customerRows);
        setLogs(logRows.items);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load risk list");
      }
    }
    load();
  }, [band, logsOnly]);

  useEffect(() => {
    if (!logsOnly) return;
    const streamBase = import.meta.env.VITE_API_DASHBOARD_URL ?? "http://localhost:8001";
    const source = new EventSource(`${streamBase}/logs/risk/stream`);
    source.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as RiskLogItem;
        setLogs((prev) => {
          const next = [parsed, ...prev];
          return next.slice(0, 250);
        });
      } catch (_) {
        // no-op
      }
    };
    return () => source.close();
  }, [logsOnly]);

  const likelyDefaults = useMemo(
    () => logs.filter((l) => l.risk_band === "RED").sort((a, b) => b.pd_ensemble - a.pd_ensemble).slice(0, 20),
    [logs]
  );

  return (
    <div className="dashboard-view">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Intervention Queue</p>
          <h1>{logsOnly ? "Live Risk Logs" : "Party Risk List"}</h1>
          <p className="hero-subtitle">
            {logsOnly
              ? "Real-time model logs with incoming risk signals and likely-default customers."
              : "Filter and triage customers by current risk band and default probability."}
          </p>
          <div className="chip-row">
            <span className="chip">Likely default now: {likelyDefaults.length}</span>
            <span className="chip">Log events loaded: {logs.length}</span>
          </div>
        </div>
      </section>

      {error ? <div className="alert-banner">{error}</div> : null}

      <section className="panel">
        <div className="table-toolbar">
          <h2>{logsOnly ? "Live Log Stream" : "Risk Inventory"}</h2>
          <div className="filter-row">
            {(["ALL", "GREEN", "YELLOW", "RED"] as const).map((b) => (
              <button key={b} onClick={() => setBand(b)} className={`btn ${band === b ? "btn-primary" : "btn-muted"}`}>
                {b}
              </button>
            ))}
          </div>
        </div>
        <div className="simple-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Customer</th>
                <th>PD Ensemble</th>
                <th>Risk Band</th>
                <th>Priority</th>
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
                  <td>
                    <span className={`tag ${log.priority === "critical" ? "tag-red" : log.priority === "watch" ? "tag-amber" : "tag-green"}`}>
                      {log.priority}
                    </span>
                  </td>
                  <td>{log.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {!logsOnly ? (
        <section className="panel">
          <h2>Scored Customers</h2>
          <p>Current customers from dashboard table for intervention selection</p>
          <div className="simple-table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Customer</th>
                  <th>PD</th>
                  <th>Risk Band</th>
                </tr>
              </thead>
              <tbody>
                {customers.map((c) => (
                  <tr key={c.customer_id}>
                    <td>{c.customer_id}</td>
                    <td>{c.pd_final.toFixed(4)}</td>
                    <td>
                      <span className={`tag ${c.risk_band === "RED" ? "tag-red" : c.risk_band === "YELLOW" ? "tag-amber" : "tag-green"}`}>
                        {c.risk_band}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}
