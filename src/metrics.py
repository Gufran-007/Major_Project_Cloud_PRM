"""
Error metrics.

THE POINT OF THIS FILE
----------------------
Standard MAE treats a 5-point over-prediction and a 5-point under-prediction as
equally bad. In a data centre they are not remotely equal:

    over-prediction  -> you provisioned capacity nobody used   -> wasted money
    under-prediction -> you provisioned too little             -> SLA breach

So every metric here is ALSO reported split by direction. The asymmetry is the
project's central argument; burying it inside a single averaged number would
throw away the thing being demonstrated.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _safe(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Return the full metric set for one model on one split."""
    y, p = _safe(y_true), _safe(y_pred)
    err = p - y                     # positive = over-predicted
    over, under = err > 0, err < 0

    denom = (np.abs(y) + np.abs(p)) / 2.0
    smape = np.mean(np.where(denom == 0, 0.0, np.abs(err) / np.where(denom == 0, 1, denom))) * 100

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    return {
        "MAE":        float(np.mean(np.abs(err))),
        "RMSE":       float(np.sqrt(np.mean(err ** 2))),
        "sMAPE_pct":  float(smape),
        "R2":         float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        # --- the asymmetric half of the story -------------------------------
        "MAE_over":   float(np.mean(np.abs(err[over]))) if over.any() else 0.0,
        "MAE_under":  float(np.mean(np.abs(err[under]))) if under.any() else 0.0,
        "under_rate_pct": float(under.mean() * 100),   # how often we under-provisioned
        "bias":       float(np.mean(err)),             # <0 means systematically low
    }


def skill_vs_naive(model_mae: float, naive_mae: float) -> float:
    """Fraction of the naive baseline's error removed. Negative = worse than naive.

    Report this next to raw MAE. An MAE of 3.1 means nothing on its own; "18%
    better than persistence" is a claim an examiner can actually interpret.
    """
    if naive_mae <= 0:
        return float("nan")
    return (naive_mae - model_mae) / naive_mae * 100
