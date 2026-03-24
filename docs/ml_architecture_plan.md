# Anvaya Phase-2: Single-Model PD Pipeline Architecture

This document serves as the authoritative specification for the Anvaya Phase-2 Credit Risk Model (PD Pipeline). 

## 1. Input Data and Features
*   **Base Dataset**: Credit-risk dataset with binary `default_flag` (0/1).
*   **Feature Set**: Exactly 13 agreed features (F1–F13) are used as predictors.
*   **Constraint**: No other variables are permitted as direct model inputs.

## 2. Data Cleaning
*   **Missing Values**: Handled with documented rules (e.g., median/mode imputation or bin-specific defaults; no silent dropping).
*   **Outliers**: Capping/Winsorization is applied where necessary and documented.
*   **Format**: All 13 features are numericized and validated for the binning stage.

## 3. Binning and Weight of Evidence (WoE)
*   **Supervised Binning**: Features are grouped into risk-meaningful buckets with monotone risk trends where logically expected.
*   **Minimum Bin Size**: Mandatory constraints to ensure statistical stability.
*   **WoE Calculation**: Computed using the standard formula.
*   **State Locking**: Bin edges and associated WoE values are frozen for production consistency.

## 4. WoE Transformation Step
A reusable transformation layer maps raw inputs (F1–F13) → 13-dimensional WoE vector:
*   **Consistency**: Identical logic for Training, Validation, and Production.
*   **Fixed Order**: Documentation specifies the exact order (F1_WoE ... F13_WoE).

## 5. Single PD Model (LightGBM)
*   **Model Type**: A single LightGBM model trained on the 13 WoE features.
*   **Objective**: Maximize discrimination (AUC) while controlling for overfit.
*   **Freezing**: Once approved, model parameters, hyperparameters, and the feature sequence are version-locked.

## 6. PD Prediction
*   **Output**: Predicted Probability of Default (PD) on a 0–1 or 0–100% scale (standardized to 0–100% for Anvaya).
*   **Scale**: Consistently documented across the pipeline.

## 7. Quality Gate
The model must pass four mandatory checks on a held-out/test slice:
1.  **AUC-ROC**: Discrimination baseline.
2.  **Accuracy + Confusion Matrix**: Interpretable performance at the decision threshold.
3.  **Calibration**: Predicted PDs vs. Actual Default Rates (e.g., Brier score/Log-loss).
4.  **Population Stability Index (PSI)**: Monitor distribution drift between training and current period.

## 8. PD Calibration
A calibration layer (e.g., Isotonic Regression) ensures:
*   Overall average PD aligns with the long-run default rate.
*   Bucket-level alignment between predicted and realized defaults.

## 9. Risk Banding (Design B)
Bands are derived purely from the Final Calibrated PD. Thresholds are calculated based on the data distribution:
*   **GREEN**: Low risk.
*   **MEDIUM**: Moderate risk.
*   **HIGH**: Significant risk.
*   **Policy**: Banding drives intervention types (Limits, Messaging, Collections).

## 10. SHAP Explainability
*   **Logic**: TreeSHAP values computed on the 13 WoE inputs.
*   **Output**: Per-customer risk drivers (positive and negative contributors) for RM-level transparency.

## 11. Governance and Versioning
*   **Freezing**: WoE tables, model binary, and hyperparameters are versioned together.
*   **Consistency**: Production scoring is a bit-perfect match for the offline evaluation flow.
