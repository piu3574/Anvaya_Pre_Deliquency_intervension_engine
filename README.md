# Anvaya Pre-Delinquency (PD) Engine v2.0.0

A production-grade credit risk scoring engine utilizing a simple-averaging ensemble of XGBoost and LightGBM models, calibrated for real-time monitoring and dashboard analytics.

## 🏗️ Repository Structure

### 🌐 API Layer
- **`api/main.py`**: FastAPI server for **Operational Real-time Scoring**. 
- **`api/dashboard_app.py`**: Flask server for **Dashboard Analytics & Statistics**.
- **`api/supabase_client.py`**: Centralized Supabase connection management.

### 🧠 Model & Training
- **`modeltraining/train_ensemble.py`**: Refined 2.0.0 training pipeline (Balanced Portfolio).
- **`modeltraining/gen_dashboard_data.py`**: Generates the 30k row "Source of Truth" CSV.
- **`modeltraining/upload_dashboard.py`**: Syncs the test split to Supabase.
- **`modeltraining/optimize_thresholds.py`**: Data-driven risk spike analysis tool.
- **`modeltraining/diagnose_distribution.py`**: Calibration and distribution auditor.

### 📊 Data & Scripts
- **`dataset/`**: Contains the 100k balanced portfolio and dashboard source files.
- **`sql/`**: PostgreSQL schema definitions for Dashboard and Operational tables.
- **`utils/`**: Helper scripts for database clearing and ID generation.

### 📜 Documentation
- **`docs/ml_architecture_plan.md`**: Technical roadmap and model logic.
- **`docs/system_audit_report.md`**: Full system health check and verification notes.

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have `python 3.9+` and the following packages:
```bash
pip install fastapi flask pydantic supabase xgboost lightgbm pandas scikit-learn
```

### 2. Launch APIs
**Operational Engine (Port 8000):**
```bash
uvicorn api.main:app --reload --port 8000
```

**Dashboard Backend (Port 8001):**
```bash
python api/dashboard_app.py
```

### ⚡ Risk Thresholds (Refined 2.0.0)
- 🟢 **GREEN**: PD < 15% (Targeted Default Rate ~1%)
- 🟡 **ORANGE**: 15% <= PD < 50% (Monitor Zone)
- 🔴 **RED**: PD >= 50% (Targeted Default Rate ~36%)
