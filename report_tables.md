# Phase-III report tables, filled from C:\Users\Gufran\Documents\MyProject\cloud-prm

Result folders found:
  - outputs  (residual window: 48)  [by_segment, per_machine, policy_comparison, policy_per_machine, summary]
  - outputs\out_w12  (residual window: 12)  [by_segment, per_machine, policy_comparison, policy_per_machine, summary]
  - outputs\out_w192  (residual window: 192)  [by_segment, per_machine, policy_comparison, policy_per_machine, summary]
  - outputs\out_w24  (residual window: 24)  [by_segment, per_machine, policy_comparison, policy_per_machine, summary]
  - outputs\out_w48  (residual window: 48)  [by_segment, per_machine, policy_comparison, policy_per_machine, summary]
  - outputs\out_w96  (residual window: 96)  [by_segment, per_machine, policy_comparison, policy_per_machine, summary]

Using **outputs** as the default run for Tables 4, 5, 6, 8, 9.

### Table 4 - Policy calibration
_mean over 12 machines and 4 provisioning delays_

| Policy | Parameter calibrated | Value selected (validation) | Violation rate achieved (test) % | Target % |
|---|---|---|---|---|
| Predictive | Margin multiplier k | 2 (range 1.5–3) | 4.99 | 5.0 |
| Reactive | Scale-up threshold | 0.9 (range 0.5–0.9) | 5.57 | 5.0 |

### Table 4a - calibration by provisioning delay (supporting detail)

| Delay | Policy | Value selected | Violation achieved (test) % | Target % |
|---|---|---|---|---|
| 0 | Predictive | 2 (range 1.5–2) | 3.58 | 5.0 |
| 0 | Reactive | 0.9 (range 0.8–0.9) | 4.09 | 5.0 |
| 1 | Predictive | 2 (range 2–2.5) | 4.36 | 5.0 |
| 1 | Reactive | 0.9 (range 0.75–0.9) | 4.62 | 5.0 |
| 3 | Predictive | 2.5 (range 2–3) | 5.11 | 5.0 |
| 3 | Reactive | 0.9 (range 0.5–0.9) | 5.75 | 5.0 |
| 6 | Predictive | 3 (range 2.5–3) | 6.89 | 5.0 |
| 6 | Reactive | 0.9 (range 0.5–0.9) | 7.81 | 5.0 |

### Table 5 - Predictive against reactive allocation

| Delay | Policy | SLA violation % | Utilisation % | Idle capacity | Energy proxy (Wh) | Cost J |
|---|---|---|---|---|---|---|
| 0 | Predictive | 3.58 | 84.09 | 4.72 | 5503.0 | 4.90 |
| 0 | Reactive | 4.09 | 69.00 | 11.53 | 6031.6 | 11.74 |
| 1 | Predictive | 4.36 | 83.16 | 5.05 | 5520.5 | 5.26 |
| 1 | Reactive | 4.62 | 67.53 | 12.33 | 6088.8 | 12.56 |
| 3 | Predictive | 5.11 | 79.90 | 6.24 | 5602.0 | 6.49 |
| 3 | Reactive | 5.75 | 64.50 | 13.75 | 6189.9 | 14.04 |
| 6 | Predictive | 6.89 | 77.47 | 7.21 | 5658.9 | 7.55 |
| 6 | Reactive | 7.81 | 62.78 | 14.82 | 6253.7 | 15.21 |

### Table 6 - Effect of provisioning delay on the predictive advantage

| Provisioning delay | Utilisation gain (pp) | Idle capacity reduction | Energy proxy change % | Cost J reduction |
|---|---|---|---|---|
| 0 intervals | +15.08 | 40.9% of reactive | -8.76 | 58.3% lower |
| 1 interval (5 min) | +15.63 | 40.9% of reactive | -9.33 | 58.1% lower |
| 3 intervals (15 min) | +15.40 | 45.3% of reactive | -9.50 | 53.8% lower |
| 6 intervals (30 min) | +14.69 | 48.6% of reactive | -9.51 | 50.3% lower |

### Table 7 - Residual-window sensitivity
_predictive policy, averaged over machines and delays; source: calibrated predictive policy_

| Residual window | Mean margin | SLA violation % | Utilisation % | Cost J |
|---|---|---|---|---|
| 12 intervals (1 h) | 5.60 | 5.61 | 80.49 | 6.31 |
| 24 intervals (2 h) | 5.49 | 5.20 | 80.86 | 6.17 |
| 48 intervals (4 h) | 5.39 | 4.99 | 81.15 | 6.05 |
| 96 intervals (8 h) | 5.36 | 5.04 | 81.22 | 6.03 |
| 192 intervals (16 h) | 5.33 | 5.21 | 81.35 | 6.01 |

- lowest violation rate: window 48 (4.99%)
- lowest cost J: window 192 (6.01)
- the two criteria disagree, so the choice of window depends on the cost weights - see Table 7a

### Table 7a - which residual window wins as an SLA breach gets dearer
_alpha = 1; the project's configured beta is 5_

| beta (price of one breach, in units of idle capacity) | Window with the lowest J | J at that window | Its violation % | Its idle capacity |
|---|---|---|---|---|
| 0 | 192 intervals (16 h) | 5.75 | 5.21 | 5.75 |
| 1 | 192 intervals (16 h) | 5.80 | 5.21 | 5.75 |
| 2 | 192 intervals (16 h) | 5.85 | 5.21 | 5.75 |
| 5 | 192 intervals (16 h) | 6.01 | 5.21 | 5.75 |
| 10 | 192 intervals (16 h) | 6.27 | 5.21 | 5.75 |
| 20 | 96 intervals (8 h) | 6.79 | 5.04 | 5.78 |
| 50 | 48 intervals (4 h) | 8.29 | 4.99 | 5.80 |
| 100 | 48 intervals (4 h) | 10.79 | 4.99 | 5.80 |

- window 48 overtakes window 192 on cost once beta exceeds 22.8, i.e. once one SLA breach is priced above about 23 units of idle capacity

### Table 8 - One fixed policy, different forecasters driving it
_allocation policy and margin rule held constant; only the forecaster is swapped_

| Forecaster driving the fixed policy | Forecast MAE | Skill vs naive % | SLA violation % | Utilisation % |
|---|---|---|---|---|
| Naive persistence | 1.9364 | 0.00 | 8.78 | 87.55 |
| Seasonal naive | 7.4968 | -288.44 | 22.73 | 71.06 |
| ARIMA | 2.2180 | -14.67 | 8.11 | 86.18 |
| Ridge regression | 1.8678 | 3.40 | 9.22 | 88.32 |
| Gradient boosting | 1.9936 | -3.26 | 7.30 | 86.94 |

### Table 9 - Computational cost
_a 5-minute interval is 300,000 ms; the verdict column scales one-row inference to a 10,000-machine estate_

| Component | Training time (s) | Inference (ms per row) | Within a 5-minute interval? |
|---|---|---|---|
| Naive persistence | 0.000 | 0.0001 | Yes - 0.0% of the budget at 10,000 machines |
| Seasonal naive | 0.000 | 0.0002 | Yes - 0.0% of the budget at 10,000 machines |
| ARIMA | 0.353 | 11.8552 | Yes - 39.5% of the budget at 10,000 machines |
| Ridge regression | 0.022 | 0.0011 | Yes - 0.0% of the budget at 10,000 machines |
| Gradient boosting | 1.270 | 0.0042 | Yes - 0.0% of the budget at 10,000 machines |

---
Any cell reading [no data] has nothing behind it in the CSVs. Leave it empty in the report or say in the text why it is absent - do not fill it by hand.
