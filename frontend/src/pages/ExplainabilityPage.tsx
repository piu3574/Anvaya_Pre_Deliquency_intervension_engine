import { FormEvent, useState } from "react";
import { getScoreByCustomer } from "../api/client";

interface Driver {
  feature: string;
  direction: "up" | "down";
  impact: number;
}

interface ScoreResponse {
  customer_id: string;
  pd_final: number;
  risk_band: string;
  top_drivers: Driver[];
  model_version: string;
  timestamp: string;
}

export default function ExplainabilityPage() {
  const [customerId, setCustomerId] = useState("");
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!customerId.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = (await getScoreByCustomer(customerId.trim())) as ScoreResponse;
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to score customer");
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="dashboard-view">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Model Transparency</p>
          <h1>Explainability Console</h1>
          <p className="hero-subtitle">Fetch live score + top drivers for any customer and inspect why default risk is rising.</p>
        </div>
      </section>

      <section className="panel">
        <form onSubmit={onSubmit} className="explain-form">
          <input
            className="text-input"
            value={customerId}
            onChange={(e) => setCustomerId(e.target.value)}
            placeholder="Enter customer ID"
          />
          <button className="btn btn-primary" type="submit" disabled={loading}>
            {loading ? "Scoring..." : "Get Explainability"}
          </button>
        </form>
        {error ? <div className="alert-banner mt-12">{error}</div> : null}
      </section>

      {result ? (
        <section className="panel">
          <h2>Live Score Result</h2>
          <div className="chip-row">
            <span className="chip">Customer: {result.customer_id}</span>
            <span className="chip">PD: {result.pd_final.toFixed(4)}</span>
            <span className="chip">Risk: {result.risk_band}</span>
            <span className="chip">Model: {result.model_version}</span>
          </div>

          <div className="simple-table-wrap mt-12">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Direction</th>
                  <th>Impact</th>
                </tr>
              </thead>
              <tbody>
                {result.top_drivers.map((driver) => (
                  <tr key={driver.feature}>
                    <td>{driver.feature}</td>
                    <td>
                      <span className={`tag ${driver.direction === "up" ? "tag-red" : "tag-green"}`}>{driver.direction}</span>
                    </td>
                    <td>{driver.impact.toFixed(6)}</td>
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
