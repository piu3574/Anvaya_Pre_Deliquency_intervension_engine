import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getModelHealthSummary, getModelHealthTimeline } from "../api/client";
import type { ModelHealthSummary, ModelHealthTimelinePoint } from "../types";

export default function ModelHealthPage() {
  const [summary, setSummary] = useState<ModelHealthSummary | null>(null);
  const [timeline, setTimeline] = useState<ModelHealthTimelinePoint[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setError(null);
      try {
        const [summaryData, timelineData] = await Promise.all([getModelHealthSummary(), getModelHealthTimeline(24)]);
        setSummary(summaryData);
        setTimeline(timelineData);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load model health");
      }
    }
    load();
  }, []);

  const kpis = [
    { title: "AUC", value: summary?.auc != null ? summary.auc.toFixed(3) : "N/A" },
    { title: "Mean PD", value: summary ? `${(summary.mean_pd * 100).toFixed(2)}%` : "N/A" },
    {
      title: "Calibration Error",
      value: summary?.calibration_error != null ? `${(summary.calibration_error * 100).toFixed(2)}%` : "N/A"
    },
    { title: "Drift Proxy", value: summary ? summary.drift_proxy.toFixed(2) : "N/A" },
    { title: "Events Last Hour", value: summary ? String(summary.events_last_hour) : "N/A" },
    { title: "Sample Size", value: summary ? String(summary.sample_size) : "N/A" }
  ];

  return (
    <div className="dashboard-view">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Monitoring and Drift</p>
          <h1>Model Health</h1>
          <p className="hero-subtitle">Track calibration, discrimination, drift proxies, and risk mix using live production data.</p>
          <div className="chip-row">
            <span className="chip">RED rate: {summary ? `${summary.red_rate.toFixed(2)}%` : "N/A"}</span>
            <span className="chip">YELLOW rate: {summary ? `${summary.yellow_rate.toFixed(2)}%` : "N/A"}</span>
            <span className="chip">GREEN rate: {summary ? `${summary.green_rate.toFixed(2)}%` : "N/A"}</span>
          </div>
        </div>
      </section>

      {error ? <div className="alert-banner">{error}</div> : null}

      <section className="kpi-grid kpi-grid-6">
        {kpis.map((item) => (
          <article key={item.title} className="kpi-card">
            <div className="kpi-header">
              <p>{item.title}</p>
            </div>
            <h3>{item.value}</h3>
          </article>
        ))}
      </section>

      <section className="panel panel-chart">
        <h2>Health Timeline (24h)</h2>
        <p>Average PD and RED event concentration per hour</p>
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeline} margin={{ top: 14, right: 18, bottom: 4, left: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#d7deec" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 11, fill: "#76839c" }}
                tickFormatter={(v) => new Date(v).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
              />
              <YAxis yAxisId="left" tick={{ fontSize: 11, fill: "#76839c" }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: "#76839c" }} />
              <Tooltip labelFormatter={(v) => new Date(v).toLocaleString()} />
              <Line yAxisId="left" type="monotone" dataKey="avg_pd" stroke="#2f6aea" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="red_events" stroke="#d14343" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>
    </div>
  );
}
