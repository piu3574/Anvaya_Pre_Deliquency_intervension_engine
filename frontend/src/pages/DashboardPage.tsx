import { useEffect, useMemo, useState } from "react";
import { AlertTriangle, ArrowRight, FileWarning, PlayCircle, Sigma, Users } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getCustomers, getDashboardStats, getPipelineStatus, getRiskLogs } from "../api/client";
import type { ChartBin, CustomerRow, DashboardStatsResponse, PipelineStatus, QueueItem, RiskLogItem } from "../types";

function toQueue(customers: CustomerRow[]): QueueItem[] {
  return customers
    .filter((c) => c.risk_band === "RED" || c.risk_band === "YELLOW")
    .sort((a, b) => b.pd_final - a.pd_final)
    .slice(0, 6)
    .map((c, index) => ({
      id: c.customer_id,
      score: Number(c.pd_final.toFixed(2)),
      band: c.risk_band,
      elapsed: `${(index + 1) * 4}m`
    }));
}

function toDistribution(customers: CustomerRow[]): ChartBin[] {
  const bins = new Map<string, number>();
  for (let i = 0; i <= 10; i += 1) bins.set((i / 10).toFixed(1), 0);
  customers.forEach((row) => {
    const bucket = Math.min(1, Math.max(0, Math.floor(row.pd_final * 10) / 10)).toFixed(1);
    bins.set(bucket, (bins.get(bucket) ?? 0) + 1);
  });
  return Array.from(bins.entries()).map(([bucket, value]) => ({ bucket, value }));
}

export default function DashboardPage() {
  const [stats, setStats] = useState<DashboardStatsResponse | null>(null);
  const [customers, setCustomers] = useState<CustomerRow[]>([]);
  const [riskLogs, setRiskLogs] = useState<RiskLogItem[]>([]);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [statsRes, customerRes, riskLogsRes, pipelineRes] = await Promise.all([
          getDashboardStats(),
          getCustomers(200, 0),
          getRiskLogs(20, 0),
          getPipelineStatus()
        ]);
        setStats(statsRes);
        setCustomers(customerRes);
        setRiskLogs(riskLogsRes.items);
        setPipelineStatus(pipelineRes);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unable to fetch dashboard data";
        setError(message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const distribution = useMemo(() => toDistribution(customers), [customers]);
  const queue = useMemo(() => toQueue(customers), [customers]);

  const kpis = [
    {
      title: "Parties Analyzed",
      value: stats?.sample_count ?? 0,
      hint: "latest run",
      subline: "scored in current cache",
      icon: Users
    },
    {
      title: "High-Risk Alerts",
      value: customers.filter((c) => c.risk_band === "RED").length,
      hint: "above threshold",
      subline: "requires triage attention",
      icon: AlertTriangle
    },
    {
      title: "Cases Open",
      value: queue.length,
      hint: "active queue",
      subline: "investigations in progress",
      icon: FileWarning
    },
    {
      title: "Model Drift",
      value: `${Math.max(0.2, Math.abs((stats?.pd_final_stats.max ?? 0) - (stats?.pd_final_stats.mean ?? 0))).toFixed(1)}%`,
      hint: "monitoring",
      subline: "within threshold",
      icon: Sigma
    }
  ];

  return (
    <div className="dashboard-view">
      <section className="hero-panel hero-panel-soft">
        <div>
          <p className="eyebrow">AML Intelligence Cockpit</p>
          <h1>Operational Risk Command Center</h1>
          <p className="hero-subtitle">
            Track risk patterns, monitor model outcomes, and prioritize intervention from one command interface.
          </p>
          <div className="chip-row">
            <span className="chip">Pipeline: {pipelineStatus?.state ?? "unknown"}</span>
            <span className="chip">Dataset: dashboard_customers</span>
            <span className="chip">{customers.length} transactions processed</span>
          </div>
        </div>
        <div className="hero-actions">
          <button className="btn btn-primary">
            <PlayCircle size={16} />
            Run Pipeline
          </button>
          <button className="btn btn-muted">
            Open Risk Queue
            <ArrowRight size={16} />
          </button>
        </div>
      </section>

      {error && <div className="alert-banner">Backend unreachable: {error}</div>}

      <section className="kpi-grid">
        {kpis.map((kpi) => (
          <article className="kpi-card bank-kpi-card" key={kpi.title}>
            <div className="kpi-header">
              <p>{kpi.title}</p>
              <kpi.icon size={16} />
            </div>
            <h3>{loading ? "..." : kpi.value}</h3>
            <span>{kpi.hint}</span>
            <small>{kpi.subline}</small>
          </article>
        ))}
      </section>

      <section className="dashboard-bottom-grid">
        <article className="panel panel-chart">
          <div className="panel-heading-row">
            <div>
              <h2>Risk Score Distribution</h2>
              <p>Population spread across low, medium, and high-risk cohorts</p>
            </div>
            <button className="btn btn-muted">Monthly</button>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distribution} margin={{ top: 14, right: 18, bottom: 4, left: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d7deec" vertical={false} />
                <XAxis dataKey="bucket" tick={{ fontSize: 11, fill: "#76839c" }} />
                <YAxis tick={{ fontSize: 11, fill: "#76839c" }} />
                <Tooltip cursor={{ fill: "rgba(47,106,234,0.08)" }} />
                <Bar dataKey="value" fill="#2f6aea" radius={[6, 6, 0, 0]} maxBarSize={38} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </article>

        <article className="panel queue-panel">
          <div className="queue-header">
            <div>
              <h2>High-Risk Queue</h2>
              <p>Recent model alerts requiring attention</p>
            </div>
            <button className="link-btn">View all</button>
          </div>

          <div className="queue-list">
            {queue.length === 0 && !loading ? <p className="empty-state">No high-risk parties in queue.</p> : null}
            {queue.map((item) => (
              <div className="queue-item" key={item.id}>
                <div>
                  <strong>{item.id}</strong>
                  <p>{item.band === "RED" ? "Escalate immediately" : "Watchlist"}</p>
                </div>
                <div className="queue-score">
                  <span>{item.score}</span>
                  <em>{item.elapsed}</em>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <h2>Live Risk Logs</h2>
        <p>Latest real-time scoring outcomes with likely defaulted customers</p>
        <div className="simple-table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Customer</th>
                <th>PD</th>
                <th>Band</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {riskLogs.map((log) => (
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
