# Cloud-prm — Predictive Resource Management for Cloud Data Centres

B.Tech major project, School of Computer Science and Engineering,
IILM University, Greater Noida. Session 2026–27.

Cloud data centres commit capacity based on what has already happened. Commit
too much and servers sit idle drawing power; commit too little and the service
breaks its agreement. Reactive autoscalers only act once a threshold has been
crossed — by which time demand has already moved, and new capacity takes minutes
to come online.

So we forecast demand a few intervals ahead and allocate from the forecast. The
part we actually care about is different from most work in this area: **we
measure whether a better forecast produces a better allocation decision**, which
the literature generally assumes rather than tests.

---

## Quick start — 4 commands, about 3 minutes

No dataset download. No API key. No configuration.

```bash
git clone https://github.com/Gufran-007/Major_Project_Cloud-PRM.git
cd cloud-prm
pip install -r requirements.txt
python run.py --demo --policy
```

That trains four forecasters, replays two allocation policies at four
provisioning delays, and writes result CSVs into `outputs/`.

Verify the install separately with `python tests/test_smoke.py` — 11 tests,
about five seconds.

---

## What data does this use?

**By default: synthetic traces, generated on the fly. Nothing is downloaded.**

`src/synthetic.py` builds each virtual machine's CPU series from an explicit
daily sine wave, a weekly term, an AR(1) drift so consecutive samples correlate
the way real ones do, injected bursts, and a few artificial gaps so the cleaning
code has something to handle.

That is deliberate, for two reasons. It means anyone can clone this and have it
running in three minutes. And because the structure is known by construction, a
result that looks wrong on synthetic data points at a bug in our code rather
than at something odd in the dataset — which makes it a debugging instrument,
not a shortcut.

**Current published results in this README are from synthetic traces.**

### Switching to the real GWA-T-12 Bitbrains traces

The loader is implemented (`src/data.py` handles the Bitbrains column names,
semicolon separators and UNIX timestamps). It is simply switched off. To use it:

1. Download from <http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains>
2. Unzip so you have a folder of per-VM `.csv` files
3. In `config/default.yaml` set:
   ```yaml
   use_synthetic: false
   data_dir: "data/bitbrains/fastStorage"
   ```
4. `python run.py --config config/default.yaml --policy`

Validation on the real traces is pending at the time of writing.

---

## How it fits together

One command, one config file, data flowing through modules in order:

```
run.py
  └─ config/default.yaml        every tunable number lives here
       │
       ├─ src/data.py           get traces  (synthetic  OR  real Bitbrains)
       ├─ src/features.py       turn history into a table of numbers
       ├─ src/splits.py         cut it by time: 60% train / 20% val / 20% test
       ├─ src/models.py         train the forecasters
       ├─ src/metrics.py        score them, split by over/under prediction
       ├─ src/policy.py         turn a forecast into a capacity commitment
       └─ src/simulator.py      replay it with a provisioning delay
            │
            └─ outputs/*.csv    results
```

Nothing is hard-coded in the source. An experiment is fully described by
`config/default.yaml` plus the random seed, and the config actually used is
written into `outputs/` alongside the results so the two cannot drift apart.

---

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

Synthetic traces, 12 machines, ~3,950 intervals each, 71 features, 60/20/20
chronological split, one-step horizon. Means across machines on the test split.

| Model             | MAE   | RMSE  | R²    | Under-predicts | SLA violation |
| ----------------- | ----- | ----- | ----- | -------------- | ------------- |
| Ridge regression  | 1.868 | 2.683 | 0.961 | 51.1%          | 9.22%         |
| Naive persistence | 1.936 | 2.800 | 0.957 | 48.1%          | 8.78%         |
| Gradient boosting | 1.994 | 2.840 | 0.956 | 44.6%          | 7.30%         |
| Seasonal naive    | 7.497 | 9.470 | 0.519 | 47.5%          | 22.73%        |

**The two rankings come out exactly reversed.** By error: ridge, naive,
gradient boosting. By SLA violation rate: gradient boosting, naive, ridge. The
most accurate model is the worst one to provision from.

The bias column explains why. Ridge has mean bias −0.071, leans low, and
under-predicts 51.1% of the time. Gradient boosting has bias +0.208, leans high,
and under-predicts only 44.6%. Under-prediction is what breaches the SLA.

Two honest notes. Ridge removes only 3.4% of the naive baseline's error, and
gradient boosting is 3.3% _worse_ than naive — at a five-minute horizon,
persistence is a strong baseline. Gradient boosting also costs 1.31s to train
against ridge's 0.02s, sixty times the cost for worse accuracy.

The inversion survives splitting machines by predictability: the same ordering
appears in both the predictable and volatile groups.

### Predictive vs reactive allocation

Both policies calibrated on validation to the same violation target, then run
over identical test traces.

| Delay | Predictive viol. | Reactive viol. | Predictive util. | Reactive util. |
| ----- | ---------------- | -------------- | ---------------- | -------------- |
| 0     | 3.58%            | 4.09%          | 84.1%            | 69.0%          |
| 1     | 4.36%            | 4.62%          | 83.2%            | 67.5%          |
| 3     | 5.11%            | 5.75%          | 79.9%            | 64.5%          |
| 6     | 6.89%            | 7.81%          | 77.5%            | 62.8%          |

Predictive allocation is better on **both** axes at every delay — fewer
violations and higher utilisation at the same time. It holds 41–49% of the idle
capacity the reactive controller holds, and the energy proxy is about 9% lower
throughout.

**The advantage does not grow with the provisioning delay.** The utilisation gap
stays between 14.7 and 15.6 percentage points across all four delays. Both
policies degrade roughly in parallel as the delay lengthens. We had expected the
gap to widen; it did not, and we report it as observed.

All figures above are reproducible with `python run.py --policy` (seed 42).

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
