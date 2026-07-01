# 🛡️ Anvaya — Pre-Delinquency Intervention Engine

> **An AI-powered early-warning platform that predicts loan defaults *before* they happen, enabling financial institutions to proactively intervene and reduce credit risk.**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)
![Flask](https://img.shields.io/badge/Flask-black?style=for-the-badge&logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-Ensemble-success?style=for-the-badge)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Supabase-blue?style=for-the-badge&logo=postgresql)
![TypeScript](https://img.shields.io/badge/Frontend-TypeScript-3178C6?style=for-the-badge&logo=typescript)

</p>

---

# 📌 Overview

Traditional credit risk systems identify borrowers only after they miss repayments. By then, lenders have already incurred financial risk.

**Anvaya** is an AI-powered **Pre-Delinquency Intervention Engine** that predicts the **Probability of Default (PD)** before delinquency occurs, allowing banks and financial institutions to take proactive actions such as reminders, restructuring, or personalized interventions.

Instead of reacting to defaults, Anvaya enables organizations to **prevent them.**

---

# 🎯 Key Features

- 📊 Real-time Probability of Default Prediction
- 🤖 Ensemble Machine Learning Model (XGBoost + LightGBM)
- 🚦 Intelligent Risk Segmentation (Green / Orange / Red)
- ⚡ FastAPI-based Prediction API
- 📈 Interactive Dashboard (Flask + TypeScript)
- 🧠 Explainable AI Predictions
- ☁️ PostgreSQL + Supabase Integration
- 📉 Portfolio Risk Analytics
- 🔍 Data-driven Threshold Optimization
- 📦 Modular & Scalable Architecture

---

# 💡 Inspiration

Banks typically detect loan defaults after borrowers miss payments, leaving limited opportunity for preventive action.

Our vision was to shift from **reactive credit risk management** to **proactive intervention** by identifying high-risk borrowers before delinquency occurs.

Anvaya empowers lenders to make informed decisions using predictive analytics and explainable machine learning.

---

# 🚀 What Anvaya Does

The platform provides an end-to-end credit risk intelligence system.

✔ Predicts Probability of Default for every borrower

✔ Classifies borrowers into:

- 🟢 Green (Low Risk)
- 🟡 Orange (Medium Risk)
- 🔴 Red (High Risk)

✔ Serves predictions through FastAPI

✔ Displays portfolio insights on a live dashboard

✔ Stores predictions in PostgreSQL (Supabase)

✔ Provides explainable predictions for risk analysts

---

# 🏗️ System Architecture

```
                    +----------------------+
                    | Borrower Information |
                    +----------+-----------+
                               |
                               v
                    Feature Engineering
                               |
                               v
          +-------------------------------------+
          | XGBoost + LightGBM Ensemble Model   |
          +----------------+--------------------+
                           |
                    Probability of Default
                           |
        +------------------+------------------+
        |                                     |
        v                                     v
 FastAPI Prediction API             Flask Dashboard
        |                                     |
        +------------------+------------------+
                           |
                    PostgreSQL (Supabase)
                           |
                           v
                 Portfolio Risk Analytics
```

---

# 📂 Project Structure

```
Anvaya_Pre_Deliquency_intervension_engine/

├── api/
│   ├── main.py
│   ├── dashboard_app.py
│   └── supabase_client.py
│
├── modeltraining/
│   ├── train_ensemble.py
│   ├── gen_dashboard_data.py
│   ├── optimize_thresholds.py
│   └── diagnose_distribution.py
│
├── dataset/
│
├── feature_engineering/
│
├── explainability/
│
├── frontend/
│
├── sql/
│
├── tests/
│
├── docs/
│   ├── ANVAYA.pdf
│   ├── ml_architecture_plan.md
│   └── system_audit_report.md
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Machine Learning Pipeline

1. Data Collection
2. Feature Engineering
3. Data Cleaning
4. Ensemble Model Training
5. Probability Calibration
6. Threshold Optimization
7. Explainability Generation
8. FastAPI Model Serving
9. Dashboard Analytics
10. Continuous Monitoring

---

# 🚦 Risk Classification

| Risk Zone | Probability of Default | Action |
|------------|-----------------------|---------|
| 🟢 Green | PD < 15% | Safe |
| 🟡 Orange | 15% ≤ PD < 50% | Monitor |
| 🔴 Red | PD ≥ 50% | Immediate Intervention |

---

# ⚙️ Technology Stack

## Machine Learning

- XGBoost
- LightGBM
- Scikit-Learn
- Pandas
- NumPy

## Backend

- FastAPI
- Flask
- Pydantic

## Database

- PostgreSQL
- Supabase

## Frontend

- TypeScript

## Testing

- Pytest

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/piu3574/Anvaya_Pre_Deliquency_intervension_engine.git

cd Anvaya_Pre_Deliquency_intervension_engine
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Start Prediction API

```bash
uvicorn api.main:app --reload --port 8000
```

---

## Start Dashboard Backend

```bash
python api/dashboard_app.py
```

---

## Start Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# 🌐 API

### Predict Risk

```
POST /predict
```

Example Request

```json
{
    "age":35,
    "income":65000,
    "loan_amount":250000
}
```

Example Response

```json
{
   "probability_default":0.28,
   "risk_zone":"ORANGE",
   "explanation":"High utilization and declining repayment behaviour"
}
```

---

# 📈 Explainability

Unlike traditional black-box models, every prediction generated by Anvaya is accompanied by an explanation describing the key factors responsible for the assigned risk score.

This enables transparency, regulatory compliance, and improved trust for financial institutions.

---

# 📊 Dashboard

The dashboard provides

- Portfolio Overview
- Risk Distribution
- Green / Orange / Red Segmentation
- Live Borrower Scores
- Risk Trends
- Portfolio Analytics

---

# 📄 Technical Documentation

Complete project documentation is available here:

## 📘 [ANVAYA Technical Documentation (PDF)](docs/ANVAYA.pdf)

The documentation includes:

- System Architecture
- ML Pipeline
- Feature Engineering
- Model Design
- Database Design
- Workflow
- Deployment
- Technical Decisions
- Future Scope

---

# 🧗 Challenges

- Building a well-calibrated Probability of Default model
- Optimizing thresholds using real distribution patterns
- Balancing explainability with prediction speed
- Designing a scalable modular architecture
- Integrating FastAPI, Flask and Supabase

---

# 🏆 Achievements

- End-to-end ML pipeline
- Ensemble learning architecture
- Real-time prediction API
- Explainable AI
- Data-driven threshold optimization
- Interactive analytics dashboard
- Modular production-ready project structure

---

# 🔮 Future Improvements

- Automated Model Retraining
- SHAP-based Explainability
- Real Credit Bureau Data Integration
- Email/SMS Risk Alerts
- Docker Deployment
- Kubernetes Support
- CI/CD Pipeline
- Cloud Deployment (AWS/Azure/GCP)

---

# 📚 Documentation

| File | Description |
|------|-------------|
| docs/ANVAYA.pdf | Complete Technical Documentation |
| docs/ml_architecture_plan.md | ML Architecture |
| docs/system_audit_report.md | System Audit |
| supabase_integration_guide.md | Supabase Setup |

---

# 👥 Team

Built with ❤️ during a Hackathon by **Team Anvaya**

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

It motivates us to continue improving the project.

---
