import { useEffect, useState } from "react";
import { getRuntimeConfig } from "../api/client";
import type { RuntimeConfig } from "../types";

export default function ConfigurationPage() {
  const [config, setConfig] = useState<RuntimeConfig | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setError(null);
      try {
        const data = await getRuntimeConfig();
        setConfig(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load runtime config");
      }
    }
    load();
  }, []);

  return (
    <div className="dashboard-view">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Runtime and Controls</p>
          <h1>Configuration</h1>
          <p className="hero-subtitle">Live runtime thresholds and service config exposed from backend configuration endpoints.</p>
        </div>
      </section>

      {error ? <div className="alert-banner">{error}</div> : null}

      <section className="kpi-grid">
        <article className="kpi-card">
          <div className="kpi-header">
            <p>GREEN &lt; threshold</p>
          </div>
          <h3>{config ? config.risk_thresholds.green_lt.toFixed(2) : "N/A"}</h3>
        </article>
        <article className="kpi-card">
          <div className="kpi-header">
            <p>RED &gt;= threshold</p>
          </div>
          <h3>{config ? config.risk_thresholds.red_gte.toFixed(2) : "N/A"}</h3>
        </article>
        <article className="kpi-card">
          <div className="kpi-header">
            <p>Model Version</p>
          </div>
          <h3>{config?.service.version ?? "N/A"}</h3>
        </article>
        <article className="kpi-card">
          <div className="kpi-header">
            <p>Risk Log Table</p>
          </div>
          <h3>{config?.service.risk_log_table ?? "N/A"}</h3>
        </article>
      </section>
    </div>
  );
}
