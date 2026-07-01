# 🛡️ Anvaya — Pre-Delinquency Intervention Engine

**An AI-powered early-warning system that predicts loan defaults *before* they happen — giving lenders a window to intervene and save at-risk borrowers.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%2B%20LightGBM-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Flask](https://img.shields.io/badge/Dashboard-Flask-black)
![Status](https://img.shields.io/badge/Status-Hackathon%20Project-purple)

---

## 💡 Inspiration

Traditional credit risk systems flag borrowers only *after* they've already missed payments — by then, it's often too late to prevent the default. We built **Anvaya** to flip that model: score every borrower's **Probability of Default (PD)** in real time, and surface early-risk signals so lending teams can intervene *before* delinquency happens, not after.

---

## 🚀 What It Does

Anvaya is an end-to-end credit risk intelligence platform that:

- 📊 **Scores every borrower** with a calibrated ML ensemble (XGBoost + LightGBM) to predict PD
- 🚦 **Segments the portfolio** into GREEN / ORANGE / RED risk zones for instant triage
- ⚡ **Serves real-time scores** via a FastAPI microservice for operational use
- 📈 **Powers a live dashboard** (Flask backend + TypeScript frontend) so risk teams can monitor portfolio health at a glance
- 🔍 **Explains every prediction** — no black-box scores, so risk analysts can see *why* a borrower was flagged
- ☁️ **Syncs to Supabase** for persistent, queryable storage of scored data

---

## 🧠 How We Built It

We approached this as a full-stack ML system, not just a model:

1. **Data & Feature Engineering** — built a balanced 100k-row synthetic credit portfolio and engineered features that capture early behavioral risk signals (`feature_engineering/`)
2. **Modeling** — trained a simple-averaging ensemble of XGBoost and LightGBM, tuned and calibrated against real default-rate distributions (`modeltraining/train_ensemble.py`)
3. **Threshold Calibration** — instead of guessing risk cutoffs, we wrote a data-driven risk-spike analyzer (`optimize_thresholds.py`) to find the PD thresholds that actually separate low, medium, and high-risk borrowers
4. **Serving Layer** — split the system into two services: a **FastAPI** engine for real-time scoring and a **Flask** backend for dashboard analytics, backed by **PostgreSQL/Supabase**
5. **Explainability** — added tooling so every score comes with a "why," critical for any real-world credit decisioning tool
6. **Frontend Dashboard** — a TypeScript dashboard to visualize risk segments, trends, and portfolio health in real time

---

## 🚦 Risk Zones (v2.0.0)

| Zone | PD Threshold | Meaning |
|------|-----------|---------|
| 🟢 **GREEN** | PD < 15% | Low risk — safe zone |
| 🟡 **ORANGE** | 15% ≤ PD < 50% | Monitor zone — early intervention candidate |
| 🔴 **RED** | PD ≥ 50% | High risk — immediate action needed |

---

## 🏗️ Architecture

```
Anvaya_Pre_Deliquency_intervension_engine/
├── api/
│   ├── main.py                 # FastAPI — real-time operational scoring
│   ├── dashboard_app.py        # Flask — dashboard analytics & statistics
│   └── supabase_client.py      # Supabase connection management
│
├── modeltraining/
│   ├── train_ensemble.py       # Ensemble training pipeline (XGBoost + LightGBM)
│   ├── gen_dashboard_data.py   # Generates the scored "source of truth" dataset
│   ├── optimize_thresholds.py  # Data-driven risk threshold calibration
│   └── diagnose_distribution.py# Model calibration & drift auditor
│
├── dataset/                    # Balanced portfolio (100k rows)
├── feature_engineering/        # Feature generation & transformation logic
├── explainability/             # Model interpretability tooling
├── frontend/                   # Risk dashboard UI (TypeScript)
├── sql/                        # PostgreSQL schema definitions
├── tests/                      # Test suite
└── docs/                       # Architecture plan & system audit report
```

---

## 🛠️ Tech Stack

**ML/Data:** XGBoost · LightGBM · scikit-learn · pandas
**Backend:** FastAPI · Flask · Pydantic
**Database:** PostgreSQL (Supabase)
**Frontend:** TypeScript
**Tooling:** pytest

---

## ⚙️ Running It Locally

```bash
# Clone
git clone https://github.com/piu3574/Anvaya_Pre_Deliquency_intervension_engine.git
cd Anvaya_Pre_Deliquency_intervension_engine

# Install dependencies
pip install -r requirements.txt

# Start the real-time scoring API
uvicorn api.main:app --reload --port 8000

# Start the dashboard backend (separate terminal)
python api/dashboard_app.py

# Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Supabase credentials are required for persistence — see `supabase_integration_guide.md`.

---

## 🧗 Challenges We Ran Into

- **Calibration over accuracy** — a model that's 90% accurate but poorly calibrated is dangerous in credit risk. We built a dedicated distribution-diagnosis tool to make sure predicted PDs actually matched real-world default rates.
- **Threshold selection** — rather than picking arbitrary cutoffs for GREEN/ORANGE/RED, we built a risk-spike analyzer to derive thresholds directly from the data.
- **Explainability vs. speed** — balancing a fast, real-time FastAPI scoring path with a separate, richer analytics path for the dashboard.

---

## 🏆 Accomplishments We're Proud Of

- Built a full ML pipeline — from synthetic data generation to a calibrated ensemble model to a live-serving API — in hackathon time
- Designed risk thresholds that are *data-derived*, not arbitrary guesses
- Shipped both an operational scoring service **and** an analytics dashboard, not just a notebook
- Added real explainability tooling instead of treating the model as a black box

---

## 🔮 What's Next

- Automated model retraining pipeline as new data comes in
- Alerting/notification system for RED-zone borrowers
- Expanded explainability (SHAP-based per-feature breakdowns) surfaced directly in the dashboard
- Support for real-world, non-synthetic credit bureau datasets

---

## 📄 Docs

- [`docs/ml_architecture_plan.md`](./docs/ml_architecture_plan.md) — technical roadmap & model design
- [`docs/system_audit_report.md`](./docs/system_audit_report.md) — system health check
- [`supabase_integration_guide.md`](./supabase_integration_guide.md) — Supabase setup guide

---

## 👥 Team

Built during a hackathon by [Team Anvaya].

---

## 📜 License

No license specified yet — consider adding one before sharing publicly.
