# Ensemble PD Model Performance Report

## Tier 1: Core Performance
| Metric | Value |
| :--- | :--- |
| **AUC-ROC** | 0.8473 |
| **KS Statistic** | 0.5578 |
| **Gini Coefficient** | 0.6947 |
| **Accuracy** | 0.8683 |
| **Precision** | 0.3850 |
| **Recall** | 0.6404 |

### Confusion Matrix (at RED threshold t=0.0816)
| | Pred 0 | Pred 1 |
| :--- | :--- | :--- |
| **Actual 0** | 16146 (TN) | 1949 (FP) |
| **Actual 1** | 685 (FN) | 1220 (TP) |

### Top 5 Feature Importance (LightGBM Split)
| feature                  |   importance |
|:-------------------------|-------------:|
| F14_active_loan_pressure |          283 |
| F12_cross_loan           |          166 |
| F3_salary_delay          |          163 |
| F1_emi_to_income         |          126 |
| F4_spend_shift           |          123 |

## Tier 2: Advanced Diagnostics
| Metric | Value |
| :--- | :--- |
| **PSI (Dev vs Test)** | 1.6359 |
| **Brier Score** | 0.0642 |

### Lift Chart (by Decile, 9=Riskiest)
|   decile |        y |
|---------:|---------:|
|        0 | 0.136483 |
|        1 | 0.183727 |
|        2 | 0.257218 |
|        3 | 0.251969 |
|        4 | 0.288714 |
|        5 | 0.383202 |
|        6 | 0.572178 |
|        7 | 0.981627 |
|        8 | 1.69029  |
|        9 | 5.25459  |

### Calibration Curve Data
| Decile | Avg Predicted PD | Observed Default Rate |
| :--- | :--- | :--- |
| 0 | 0.0393 | 0.0130 |
| 1 | 0.0405 | 0.0175 |
| 2 | 0.0416 | 0.0245 |
| 3 | 0.0425 | 0.0240 |
| 4 | 0.0432 | 0.0275 |
| 5 | 0.0447 | 0.0365 |
| 6 | 0.0492 | 0.0545 |
| 7 | 0.0611 | 0.0935 |
| 8 | 0.0936 | 0.1610 |
| 9 | 0.5186 | 0.5005 |
