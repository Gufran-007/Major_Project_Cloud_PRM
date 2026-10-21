# cloud-prm — Predictive Resource Management for Cloud Data Centres

B.Tech major project, School of Computer Science and Engineering,
IILM University, Greater Noida. Session 2026–27.

Cloud data centres commit capacity based on what has already happened. Commit
too much and servers sit idle drawing power; commit too little and the service
breaks its agreement. Reactive autoscalers only act once a threshold has been
crossed — by which time the demand has already moved, and new capacity takes
minutes to come online.

So we forecast demand a few intervals ahead and allocate from the forecast. The
part we actually care about is different from most work in this area: **we
measure whether a better forecast produces a better allocation decision**, which
the literature generally assumes rather than tests.

---

## Run it

No data download needed — a synthetic trace generator ships with the code.

```bash
pip install -r requirements.txt
python tests/test_smoke.py                                  # 11 tests
python run.py --demo --policy                               # full pipeline
python run.py --demo --ablation                             # feature study
```

Results land in `outputs/` as CSV.

To use the real GWA-T-12 Bitbrains traces, set `use_synthetic: false` in
`config/default.yaml` and point `data_dir` at the `fastStorage` folder.

---

## What's here

```
src/
  config.py            experiment configuration; unknown keys raise
  synthetic.py         trace generator with known structure
  data.py              loading, resampling, gap handling
  features.py          lag / rolling / calendar / cross-resource groups
  splits.py            chronological split, training-only scaling
  models.py            every forecaster behind one fit/predict interface
  metrics.py           error split by direction, skill score
  residuals.py         rolling-residual dispersion → safety margin
  forecast_service.py  the frozen forecaster → allocator contract
  policy.py            predictive policy + reactive threshold baseline
  simulator.py         trace replay, provisioning delay, calibration
  experiment.py        orchestration, writes result CSVs
tests/test_smoke.py    11 tests
outputs/               committed results
```

---

## Three design decisions worth explaining

**Every forecaster sits behind one interface.** The allocator receives a demand
estimate, a dispersion number, a timestamp and a model ID — never the model
itself. A policy that could inspect the model could be tuned to suit whichever
model was under test, and the comparison would stop measuring what it claims to.

**The safety margin comes from rolling residuals, not a constant.** The margin
is `k × IQR` of the forecaster's errors over a trailing window. It narrows when
the model has been reliable and widens when it hasn't. Because it depends only
on realised errors, it works identically for ARIMA and for an LSTM — which is
what stops it confounding the comparison.

**Both policies are calibrated before being compared.** A cautious policy can
always buy a lower violation rate by over-provisioning. Each policy is first
tuned on validation data to the same SLA violation target, then evaluated on
test. The question becomes: at equal service quality, which wastes less?

---

## Results so far

Synthetic traces, 6 machines, ~3,950 intervals each, 71 features, 60/20/20
chronological split, one-step horizon.

| Model | MAE | RMSE | Under-predicts | SLA violation |
|---|---|---|---|---|
| Ridge regression | 1.871 | 2.727 | 51.5% | 9.81% |
| Naive persistence | 1.929 | 2.819 | 48.4% | 8.63% |
| Gradient boosting | 1.993 | 2.809 | 45.3% | 7.29% |
| Seasonal naive | 7.717 | 9.705 | 49.0% | 23.38% |

**Ridge has the lowest error and the highest SLA violation rate. Gradient
boosting is the reverse.** A model can be more accurate on average while
under-predicting more often — and under-prediction is what breaches the
agreement. Ranking by accuracy gives a different order than ranking by
allocation outcome. That is the result this project exists to report.

Predictive vs reactive allocation, calibrated to a common violation target:

| Provisioning delay | Predictive utilisation | Reactive utilisation |
|---|---|---|
| 0 intervals | 85.3% | 68.7% |
| 3 intervals | 81.9% | 65.8% |
| 6 intervals | 78.9% | 65.2% |

Idle capacity is roughly halved at every delay.

---

## Testing

11 tests, each guarding a property whose failure would invalidate results
without causing any visible error. They have caught five real defects during
development, including one that silently turned every one-step forecast into a
two-step forecast — invisible, because it affected all models equally and left
the ranking unchanged.

---

## Honest limits

- Evaluation is by trace replay, not on production infrastructure. No number
  here is a measured saving on real hardware.
- Current results are on synthetic traces. Bitbrains validation is pending.
- Energy is a utilisation-derived proxy, not a measurement.
- VM migration, network topology and multi-tenant interference are out of scope.
- LSTM and GRU are implemented but not yet trained.
