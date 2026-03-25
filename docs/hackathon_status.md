# Anvaya: Hackathon Readiness & Gap Analysis

Anvaya is in an **exceptionally strong position** for a fintech/AI hackathon. It tackles a critical real-world problem (proactive debt management vs. reactive collections) and combines heavy-hitting ML with a highly polished, production-grade frontend. 

However, to win a hackathon, the *story* and the *demo flow* need to be flawless. Below is an analysis of why Anvaya is a winning project and the critical gaps that must be closed before the final pitch.

---

## 🌟 Why This is a Winning Hackathon Project

### 1. The Narrative is Highly Empathic & Impactful
Instead of "we built an AI to punish people who don't pay loans," the pitch is: *"We predict financial distress 4 weeks early so banks can offer payment holidays and restructuring before the customer ruins their credit score."* This hits **Social Impact, ESG, and Consumer Duty** themes perfectly.

### 2. The Tech Stack is "Heavy" (Non-Trivial)
Hackathon judges often penalize simple OpenAI API wrappers. Anvaya boasts a serious architecture:
- **Two-Stage ML Ensemble**: XGBoost (Fast Path) + LightGBM (Deep Path) + Logistic Meta-learner.
- **Statistical Rigor**: Weight of Evidence (WoE) and Information Value (IV) feature engineering.
- **Real-Time Data**: Supabase integration and Server-Sent Events (SSE) streaming.
- **Explainable AI (XAI)**: SHAP values are calculated live to generate "Reason Codes" (e.g., *F5 Auto-Debit Fails*).

### 3. The Frontend Looks Production-Ready 
Thanks to the Impeccable design upgrades (Outfit/Plus Jakarta Sans typography, glassmorphism, responsive grid), the dashboard looks like a real SaaS product, not a weekend hack.

---

## ⚠️ Critical Gaps (What to fix before the demo)

To ensure a gapless 3-minute pitch, focus on these missing pieces immediately:

### Gap 1: Realistic Mock Data
- **Problem**: The dashboard is only impressive if the data looks real. Empty charts or 100% "0.0 PD" scores will kill the demo.
- **Fix**: We need a script to populate Supabase with 100–200 heavily synthesized customer records. We must intentionally engineer a few records to trigger RED (>35% PD) and YELLOW (15-35% PD) paths so the SHAP drivers light up on the screen.

### Gap 2: The "Intervention" Step is Missing
- **Problem**: We call it an "Intervention Engine," but the dashboard currently just *detects* risk. It doesn't show the actual intervention.
- **Fix**: Add a **"Generate Action Plan" button** on the frontend that pops open a modal. When clicked, it should take the customer's SHAP drivers and pass them to an LLM (OpenAI/Gemini) to generate an empathetic script for the Relationship Manager (e.g., *"Hi John, I noticed a slight delay in your salary credits. We can offer a 30-day payment holiday to help you breathe."*). This is Stage 13 of your PRD.

### Gap 3: Missing Real-Time Trigger Simulation
- **Problem**: The PRD claims Kafka captures events like "auto-debit fail" in real-time within a 20ms SLA. Right now, our UI just runs a batch scoring job.
- **Fix**: For the demo, we should build a small "Event Injector" script. While you are pitching, you can hit a script that instantly pushes an "overdraft event" to Supabase, which the frontend picks up live, instantly triggering a RED alert popup on the dashboard.

### Gap 4: Edge Case "Vulnerable Customer" Routing
- **Problem**: The PRD mentions routing "vulnerable" people (e.g., bereavement) directly to humans. 
- **Fix**: Add a tiny visual badge in the UI. If `F10_peer_stress` or a specific distress flag is triggered, show a purple "VULNERABLE - HUMAN ROUTING ONLY" badge to prove the engine is ethically constrained.

---

## 🎯 Proposed Next Steps
1. **[Data]** Write a Python script to seed Supabase with perfect demo data.
2. **[GenAI]** Wire in a quick LLM prompt generation step in the FastAPI backend for the RM Briefs.
3. **[UI]** Build the "Generate Action Plan" modal in the frontend.
