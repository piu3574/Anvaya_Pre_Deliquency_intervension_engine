# ML Architecture Plan: Ensemble PD Pipeline (XGB + LGB)

This document specifies the authoritative version of the Anvaya Pre-Delinquency (PD) model architecture.

## 1. Overview
The system uses a **Stacked Ensemble** approach to maximize predictive power and stability. Two base models (XGBoost and LightGBM) are combined via a Logistic Regression meta-model.

## 2. Feature Engineering: 13-Feature WoE
The modeling pipeline uses a fixed set of 13 behavioral features (F1–F14, excluding F11). 
- **No IV Filtering**: All 13 features are used in all models.
- **WoE Transformation**: Features are binned and transformed into Weight of Evidence (WoE) scores based strictly on the training split.
- **Locked Binning**: Bin edges and WoE values are frozen for production consistency.

## 3. Modeling Architecture

### 3.1 Base Models
Two gradient-boosted decision tree models are trained independently on the 13 WoE features:
1.  **XGBoost (M_xgb13)**: Excellent for capturing non-linear interactions and sharp decision boundaries.
2.  **LightGBM (M_lgbm13)**: Highly efficient and robust to noise, capturing deeper hierarchical patterns.

### 3.2 Meta-Model (Logistic Ensemble with Scaling)
A regularized Logistic Regression model combines the probabilities from the base models:
- **Scaling Layer**: Input probabilities `[pd_xgb, pd_lgbm]` are z-score standardized using a `StandardScaler` (mean=0, std=1) fit on the Validation set.
- **Regularization**: L2 penalty with $C=0.1$ to ensure stability and prevent any single base model from dominating.
- **Formula**:
  - $z = \text{Scaler}([pd_{xgb}, pd_{lgbm}])$
  - $logit(s) = a + w_1 \cdot z_1 + w_2 \cdot z_2$
- **Output (PD_final)**: $\sigma(logit(s))$
- **Key Artifacts**: `ensemble_meta.pkl` and `meta_scaler.pkl`.

## 4. Training Protocol
1.  **Split**: Stratified 60% Train / 20% Validation / 20% Test.
2.  **WoE**: Fit on Train only (locked for production).
3.  **Base Models**: XGBoost and LightGBM trained on Train (13 WoE).
4.  **Meta-Model Scaling**: `StandardScaler` fitted on Validation split base PDs.
5.  **Meta-Model Training**: Logistic Regression ($C=0.1$) fitted on **scaled** Validation base PDs.
6.  **Thresholds**: Risk band cut-offs (T_G, T_Y, T_R) determined on Validation output distribution.

## 5. Risk Banding
Banding is applied only to the `pd_final` score:
- 🟢 **GREEN**: `pd_final < T_G`
- 🟡 **YELLOW**: `T_G <= pd_final < T_R`
- 🔴 **RED**: `pd_final >= T_R`

## 6. Explainability (SHAP)
For all YELLOW and RED customers, SHAP values are calculated using the **LightGBM** base model's 13 features. The Top 3-5 features are extracted as risk drivers.

## 7. Production Execution Flow
**HTTP Request → Raw Features → WoE Transform → Base Scores (XGB + LGB) → Logistic Ensemble → PD_final → Banding → SHAP → JSON Response.**

---
*Version: 2.0 (Ensemble Architecture)*
