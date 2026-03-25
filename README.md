# Anvaya: Pre-Delinquency Intervention Engine

![Dashboard Architecture](docs/Architecture_diag_stage_2.png) 
*(Note: Replace with actual screenshot of your dashboard)*

Anvaya represents a paradigm shift in banking risk management: moving from **reactive debt collections** to **proactive, empathetic pre-delinquency intervention**. By predicting financial distress 4 weeks before a customer misses a payment, banks can offer restructuring or payment holidays, preserving the customer's credit score and the bank's bottom line.

## 🚀 The Working Complete Flow

Anvaya operates cohesively across three main layers: Data Ingestion, The ML Engine, and The Frontend Dashboard.

### 1. Data Ingestion & Transformation 
*   **Live Event Streaming**: The system is designed to ingest high-velocity behavioral events (e.g., `F5: Auto-Debit Failures`, `F3: Savings Account Depletion Velocity`) via streaming platforms like Kafka.
*   **Database**: Raw feature data is instantly synced to an operational **Supabase (PostgreSQL)** database.
*   **Feature Engineering**: Raw features undergo rigorous **Weight of Evidence (WoE)** transformations based on pre-calculated Information Value (IV) bins, preparing them for the highly non-linear ML estimators.

### 2. The ML Ensemble Engine (FastAPI Backup)
When a scoring run is triggered (batch or real-time event), the FastAPI backend (`api/main.py`) executes the risk pipeline:
*   **Fast Path (XGBoost)**: Quickly handles sparse and linear features.
*   **Deep Path (LightGBM)**: Evaluates complex behavioral interactions (e.g., the interplay between dropping credit limits and rising peer stress).
*   **Meta-Learner**: A Logistic Regression layer combines the paths into a final **Probability of Default (PD)** score.
*   **Explainable AI (XAI)**: Critically, the engine uses **SHAP** to calculate the exact feature contributions for *why* the customer got that score, generating human-readable "Reason Codes".

### 3. The Relationship Manager (RM) Dashboard (React + Vite)
The intervention loop is closed by the human-in-the-loop dashboard:
*   **Live Pipeline Monitoring**: The RM can hit "Run Pipeline" to stream live logs and scoring results from the backend via **Server-Sent Events (SSE)**.
*   **Risk Triage**: Customers are instantly sorted into visually distinct bands:
    *   🟥 **RED (>35% PD)**: High risk of default within 30 days. Action required immediately.
    *   🟨 **YELLOW (15-35% PD)**: Emerging risk. Monitor closely.
    *   🟩 **GREEN (<15% PD)**: Stable.
*   **Actionable Insights**: The RM selects a high-risk customer and instantly sees the SHAP drivers (e.g., "Rising Stress Velocity"). 
*   **Empathetic Intervention (Coming Soon)**: Using the SHAP drivers, the system will use GenAI to draft an empathetic SMS or call script offering a payment holiday.

## 🛠️ Tech Stack

*   **Frontend**: React, Vite, TypeScript, Vanilla CSS (Custom Design System, Impeccable UI)
*   **Backend**: Python, FastAPI, Uvicorn, Server-Sent Events (SSE)
*   **Machine Learning**: XGBoost, LightGBM, SHAP, Scikit-learn
*   **Database**: Supabase (PostgreSQL)

## 🏃 Serving the Application

You need two terminals to run the full stack.

**1. Start the FastAPI Backend:**
```bash
cd api
# Copy .env.example to .env and add Supabase credentials
uvicorn main:app --reload --port 8000
```
*The backend exposes `/customers` and `/pipeline/run` (SSE streaming).*

**2. Start the React Dashboard:**
```bash
cd frontend
npm install
npm run dev
```
*Open `http://localhost:5173` in your browser.*
