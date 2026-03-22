# Product Requirements Document (PRD): Pre-Delinquency Intervention Engine

## 1. Overview and Problem Statement

### 1.1 Problem Statement
Banks face rising delinquency risk as economic uncertainty pressures household finances. Most institutions only intervene **after** a customer misses a payment—a point at which recovery likelihood drops sharply and collections costs rise to 15–20% of the recovered amount. Because existing systems are heavily reactive and siloed, they fail to leverage early, subtle indicators of financial distress (e.g., delayed salary credits, dwindling savings, increased reliance on lending apps).

### 1.2 The Challenge
Early indicators are subtle and dispersed across various data domains. Banks struggle to create unified, predictive models to catch these signals weeks ahead of actual delinquency. Furthermore, regulatory environments (e.g., FCA Consumer Duty) demand that any intervention be fair, well-explained (SR 11-7 / SS1/23), and sensitive to vulnerable customer profiles.

### 1.3 The Solution
An open-source, GenAI-augmented, Account Aggregator-enabled **Pre-Delinquency Intervention Engine**. The system predicts the likelihood of payment default 2–4 weeks ahead of time and triggers proactive, channel-agnostic, and deeply empathetic outreach (e.g., payment holidays, restructuring) powered by SHAP explainability.

---

## 2. Target Architecture and Tech Stack

The platform is designed to be highly scalable, utilizing a robust, locally deployable open-source technology stack.

### 2.1 Core ML & Engineering Stack
- **ML Models**: XGBoost, LightGBM (Classification/Prediction)
- **Deep Learning**: PyTorch, TensorFlow (Sequence modeling)
- **Feature Store**: Feast (Real-time feature management)
- **Orchestration**: Apache Airflow
- **Stream Processing**: Apache Kafka (Ingestion of events like auto-debit fails)
- **Model Serving**: BentoML / MLflow
- **Visualization**: Plotly / Dash
- **Storage/DB**: PostgreSQL / Local databases (Historical & Real-time scores)

---

## 3. Intervention Pipeline Implementation

The risk-scoring pipeline features a highly resilient, 14-stage execution flow triggered by real-time events.

**Stage 1: Event Trigger**  
System is awakened by an overt trigger, including an auto-debit fail, salary delay, or sudden balance drop.

**Stage 2: Kafka Capture**  
Kafka instantly captures the event tracking `customer_id`, `event_type`, and real-time stamps to prevent delays.

**Stage 3: Data Extraction**  
Parallelized extraction pulling from 5 distinct data tables asynchronously (using `asyncpg`) within an aggressive 20ms SLA.

**Stage 4: Feature Computation**  
Dask cluster computes 13 (to 17) complex raw behavioral features (F1–F17).

**Stage 5: WoE Transformation**  
The raw computed values are statistically normalized into Weight of Evidence (WoE) risk scores.

**Stage 6: IV Ranking**  
Information Value (IV) ranking identifies the Top 4 predictive WoE features while confirming the baseline of all 13.

**Stage 7: XGBoost Fast Path**  
A highly optimized XGBoost model (100 trees) uses the Top 4 WoE features for a preliminary Probability of Default (PD) score to rule out safe customers instantly.
*Threshold Switch*: If PD < 0.15, the customer exits via the **GREEN** (safe) path. If > 0.15, the record proceeds.

**Stage 8: LightGBM Deep Path**  
For customers showing risk (PD > 0.15), a LightGBM model (150 trees) evaluates the full spectrum of all 13 WoE features for a refined PD score.

**Stage 9: Ensemble Scoring**  
The engine combines the XGBoost (0.4 weight) and LightGBM (0.6 weight) predictions to generate the final PD.

**Stage 10: Quality Gate**  
Real-time checks validate AUC, accuracy, calibration, and PSI against model drift. If any metric fails, scoring falls back to a STOP alert.

**Stage 11: Risk Banding**  
Transforms the Final PD into actionable tiers:
- **YELLOW (15–35% PD)**: Flags the RM watch list.
- **RED (> 35% PD)**: Triggers an urgent 2-hour escalation response.

**Stage 12: SHAP Explainability**  
Calculates the LightGBM baseline plus exact feature contributions to explain *why* the final PD resulted as it did.

**Stage 13: Smart Alerting & GenAI Brief**  
The PD, risk band, and Top 3 SHAP drivers are routed to the RM dashboard. Combined with a lightweight LLM, this generates a contextual, empathetic RM brief.

**Stage 14: Feedback Loop**  
Monitors the real-world outcome of the intervention, tracks the success rate, and schedules monthly retraining runs to update IV scores.

---

## 4. Key Feature Signals (Taxonomy)

The engine computes critical behavioral metrics to detect stress weeks before default:

- **F1 (EMI to Income Ratio):** Fraction of monthly income consumed by total EMI debt.
- **F2 (Savings Drawdown Percentage):** Savings depletion rate against a 60-day baseline.
- **F3 (Salary Delay Days):** Shift in salary credit dates vs historical timing.
- **F4 (Spending Pattern Shift):** Cosine similarity mapping drastic reductions in discretionary 30-day spend against a 90-day baseline.
- **F5 (Auto-Debit Failures):** Counts mandates failed primarily due to insufficient funds (30 days).
- **F6 (Lending App Usage):** Activity correlating with distress borrowing (from known lending platforms).
- **F7 (Cash Hoarding Ratio):** Percentage of total spend directed towards ATM cash withdrawals.
- **F8 (Stress Velocity):** The rate of deterioration of composite financial stress (dΩ/dt solved over a 12-month trajectory).
- **F9 (Payment Timing Entropy):** Reactive cash management behavior measured via the standard deviation of payment dates.
- **F10 (Peer Cohort Stress Index):** Compares the user against their geographic/employment peer cohort to account for macro shocks.
- **F11 (Overdraft Frequency):** Days driven into negative balances (exhausted buffer).
- **F12 (Cross Loan Consistency):** Repayment variations across multiple active credit accounts.
- **F13 (Secondary Income Index):** Proportion of total income derived from non-salary sources (to curb false positives).

---

## 5. Edge Cases and Resilience Handling

1. **Gig/Freelance Workers:** Uses a specialized WoE bin table built for non-salaried users tracking variance-of-inflow to avoid bias. Adjusts decision boundaries natively.
2. **Cold-Start Customers:** Users with <90 days transaction history lack fully established trends. They run exclusively on the XGBoost fast path and use median imputation on the deep path with a +0.03 conservative uplift and AA-consent requests.
3. **Vulnerable Customer Routing:** Direct integration with bereavement or vulnerability markers ensures completely suppressed automated contacts and routes instantly to a specialist care team.
4. **Duplicate Events:** Prevents infrastructure multi-fire bugs via a Redis SHA-256 idempotency check with a 10-minute cache window.
5. **Model Mismatches:** Hard stops at inference if the feature store schema version drifts from the model's expected version, preventing silent rollout degradation.
6. **Suppression Windows:** FCA CONC 7.9 compliant cooling-off periods. Interventions are queued for 7 days post-contact unless manually overridden for RED escalations.
7. **Macro-Economic PSI Spikes:** Mass population shifts (PSI > 0.2 on ≥ 3 features) generate popup POPULATION_SHIFT alerts for manual threshold tuning by risk officers, circumventing panic scoring during economic crashes.
8. **AA Consent Revocation:** If a user revokes Account Aggregator rights, cross-bank data is instantly purged, the score recalculates silently on internal data only, and the API issues confirmation within 24 hours.
9. **Protected Attribute Safeguards:** If gender, age, or postal/race proxies enter the Top 3 SHAP parameters, interventions are downgraded automatically to 'monitor only' and audited by Model Risk (ECOA compliance).
10. **Hysteresis Flip-Flop:** Precludes alert fatigue by implementing boundary hysteresis; customers oscillating near a threshold limit (e.g., 34%-36%) must remain distinctly across the threshold for 24 hours before their color band shifts.

---

## 6. Competitive and Regulatory Moat

- **NIST AI RMF 1.0 Alignment:** Maps transparently to GOVERN, MAP, MEASURE, and MANAGE quadrants, priming the bank for certifiable safety as rigorous AI legislation approaches.
- **Account Aggregator "True Capacity":** Incorporates real-time, cross-institution financial capabilities using open banking. Unlike FICO/Experian standard scores, this reveals hidden assets located in competitor institutions.
- **FCA Consumer Duty Moat:** Integrated vulnerability-aware routing actively champions good consumer outcomes rather than exploiting pre-delinquent borrowers, aligning heavily with UK regulations.
- **Confidence Intervals (Basel IV prep):** Utilizes p10/p50/p90 PD uncertainty bands instead of primitive point measurements, driving proportional and informed RM responses.
- **Alert Hysteresis & Operational UX:** Curbs relationship manager abandonment rates directly via intelligent buffer delays on status switches—a critical UX differentiation.
- **Champion-Challenger MLOps Setup:** Operates shadow deployment testing and automated weight adjustment expected by PRA SS3/18.

---

## 7. Strategic Project Goals and Benefits
- **Substantially Lower Collections Costs**: Pre-empts severe default tracks where recovery consumes 15-20% of returned funds.
- **Boosted Recovery Rates**: Early-help actions significantly improve the likelihood a customer returns safely to good standing.
- **Stronger Brand Trust**: Proactive financial care replaces hostile collections attitudes, creating durable customer loyalty vectors.
