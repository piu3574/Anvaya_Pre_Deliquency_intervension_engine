const metrics = [
  { label: 'AUC-ROC (Ensemble)', value: '0.891', sub: 'vs. 0.843 baseline', good: true },
  { label: 'Accuracy', value: '83.4%', sub: 'holdout test set', good: true },
  { label: 'Brier Score', value: '0.112', sub: 'calibration quality', good: true },
  { label: 'PSI (latest)', value: '0.034', sub: 'within safe threshold < 0.2', good: true },
  { label: 'XGBoost AUC', value: '0.871', sub: 'fast-path model (100 trees)', good: true },
  { label: 'LightGBM AUC', value: '0.887', sub: 'deep-path model (150 trees)', good: true },
  { label: 'Last Retrained', value: 'Mar 2025', sub: 'monthly cycle', good: true },
  { label: 'Features Active', value: '13', sub: 'F1–F14 (F11 excluded)', good: true },
]

const featureImportance = [
  { name: 'F8 Stress Velocity', pct: 91 },
  { name: 'F1 EMI to Income', pct: 84 },
  { name: 'F5 Auto-Debit Fails', pct: 78 },
  { name: 'F2 Savings Drawdown', pct: 73 },
  { name: 'F3 Salary Delay', pct: 65 },
  { name: 'F4 Spend Shift', pct: 59 },
  { name: 'F6 Lending App Usage', pct: 54 },
  { name: 'F9 Payment Entropy', pct: 47 },
]

export default function ModelHealth() {
  return (
    <>
      <div className="page-header">
        <div className="page-eyebrow">Anvaya</div>
        <div className="page-title">Model Health</div>
        <div className="page-subtitle">
          Ensemble performance metrics, feature importance, and drift monitoring from the last training run.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        {metrics.map((m) => (
          <div className="card" key={m.label} style={{ padding: '18px 20px' }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
              {m.label}
            </div>
            <div style={{ fontSize: 26, fontWeight: 700, color: m.good ? 'var(--blue-700)' : 'var(--red)', letterSpacing: -0.5, marginBottom: 4 }}>
              {m.value}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{m.sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Feature Importance */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">SHAP Feature Importance</div>
              <div className="card-subtitle">Mean absolute SHAP values — LightGBM deep path</div>
            </div>
          </div>
          <div className="card-body">
            {featureImportance.map((f) => (
              <div className="driver-bar-row" key={f.name}>
                <div className="driver-bar-label">{f.name}</div>
                <div className="driver-bar-track">
                  <div className="driver-bar-fill" style={{ width: `${f.pct}%` }} />
                </div>
                <div className="driver-bar-pct">{f.pct}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Model Architecture */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Ensemble Architecture</div>
              <div className="card-subtitle">Two-stage scoring with logistic meta-learner</div>
            </div>
          </div>
          <div className="card-body">
            {[
              { stage: '1', title: 'XGBoost Fast Path', desc: '100 estimators · Top 4 WoE features · Threshold: PD < 0.15 → GREEN exit', color: 'var(--blue-500)' },
              { stage: '2', title: 'LightGBM Deep Path', desc: '150 estimators · All 13 WoE features · For customers PD > 0.15', color: 'var(--blue-700)' },
              { stage: '3', title: 'Meta Ensemble', desc: 'Logistic regressor · XGB weight 0.4, LGBM weight 0.6 · Calibrated with isotonic regression', color: 'var(--blue-900)' },
              { stage: '4', title: 'SHAP Explainability', desc: 'TreeSHAP on LightGBM · Top 3 drivers per customer · Logged to Supabase', color: 'var(--green)' },
            ].map((s) => (
              <div key={s.stage} style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                <div style={{
                  width: 28, height: 28, borderRadius: '50%', background: s.color,
                  color: 'white', fontWeight: 700, fontSize: 12, display: 'flex',
                  alignItems: 'center', justifyContent: 'center', flexShrink: 0
                }}>
                  {s.stage}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 13, color: 'var(--text-primary)', marginBottom: 2 }}>{s.title}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{s.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
