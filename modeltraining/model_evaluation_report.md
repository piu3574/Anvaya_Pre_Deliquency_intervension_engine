# Anvaya PD Engine 2.0.0 Refinement Report
## 1. Executive Summary
- **Dataset**: `hyper_realistic_portfolio_100k.csv`
- **Total Samples**: 100,000
- **Test Default Rate**: 4.09%
- **Inferred Thresholds**:
  - GREEN/YELLOW PD: **0.1297**
  - YELLOW/RED PD  : **0.2248**

## 2. Core Discrimination Metrics
| Metric | XGBoost | LightGBM | **Final Ensemble** |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | 0.6475 | 0.6488 | **0.6485** |
| **Gini** | 0.2949 | 0.2976 | **0.2970** |
| **KS Stat** | 0.2947 | 0.2934 | **0.2931** |
| **PR-AUC** | 0.1433 | 0.1413 | **0.1431** |

## 3. Class-Level Metrics (Default Class 1 @ RED Threshold)
| Metric | XGBoost | LightGBM | **Final Ensemble** |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.1581 | 0.1539 | **0.1551** |
| **Recall** | 0.3803 | 0.3762 | **0.3787** |
| **F1-Score** | 0.2234 | 0.2185 | **0.2201** |
| **Accuracy** | 0.8918 | 0.8898 | **0.8901** |

## 4. Calibration Analysis
**Hosmer-Lemeshow Test**: 
- Statistic: 3328.4366
- **p-value**: 0.0000e+00 (Lower is better calibration)

### Reliability Table
| q              |   count |   sum |    p_avg |   expected |   hl_stat |
|:---------------|--------:|------:|---------:|-----------:|----------:|
| (0.078, 0.11]  |    3001 |   100 | 0.104206 |    312.723 |   161.533 |
| (0.11, 0.115]  |    3002 |    79 | 0.112241 |    336.946 |   222.435 |
| (0.115, 0.119] |    2997 |    82 | 0.117204 |    351.259 |   233.805 |
| (0.119, 0.122] |    3000 |    82 | 0.120893 |    362.678 |   247.089 |
| (0.122, 0.126] |    3000 |    80 | 0.123844 |    371.531 |   261.091 |
| (0.126, 0.13]  |    3002 |    75 | 0.12749  |    382.725 |   283.576 |
| (0.13, 0.137]  |    2998 |   107 | 0.133183 |    399.281 |   246.829 |
| (0.137, 0.153] |    3000 |    74 | 0.144688 |    434.064 |   349.206 |
| (0.153, 0.225] |    3002 |    84 | 0.180956 |    543.231 |   473.991 |
| (0.225, 0.834] |    2998 |   465 | 0.417517 |   1251.72  |   848.883 |

## 5. Business / Portfolio Impact
- **Capture Rate @ Top 5%**: 27.44%
- **Capture Rate @ Top 10%**: 37.87%
- **Capture Rate @ Top 20%**: 44.71%
- **Lift @ Top Decile**: 3.79x

## 6. Risk-Band Distribution (Test Set)
| Band | PD Range | Count | Distribution (%) | Default Rate |
| :--- | :--- | :--- | :--- | :--- |
| **GREEN** | < 0.1297 | 17995 | 60.0% | 2.77% |
| **YELLOW** | 0.1297 - 0.2248 | 9004 | 30.0% | 2.94% |
| **RED** | >= 0.2248 | 3001 | 10.0% | 15.49% |
