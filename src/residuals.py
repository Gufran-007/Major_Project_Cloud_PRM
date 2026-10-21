"""
Rolling-residual dispersion -> the allocation safety margin.

THIS IS THE MECHANISM FROM PHASE-I SECTION 5.3
----------------------------------------------
The allocator cannot provision exactly the forecast value; that would breach
the SLA roughly half the time. It needs a margin on top. The question is where
the margin comes from.

  * A CONSTANT margin ("always add 20%") is what most autoscalers do. But it is
    an arbitrary number chosen by an administrator - the very thing the project
    criticises reactive threshold scaling for.

  * A TRUE PREDICTION INTERVAL is more principled but needs quantile regression
    or a deep ensemble for every model family, which multiplies the work.

  * WHAT WE DO: track the forecaster's own recent errors and size the margin
    from their spread. When the model has been reliable lately the margin is
    small; when it has been erratic the margin widens on its own.

The key property is that this is MODEL-AGNOSTIC. It works identically for
ARIMA and for an LSTM, so it cannot confound the swap-the-forecaster comparison.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualTracker:
    """Sliding window over recent forecast errors.

    window : how many past intervals to remember. Too short and the margin
             jitters on noise; too long and it will not widen fast enough when
             a workload turns volatile. This is a real hyperparameter - run a
             sensitivity sweep on it rather than accepting the default.
    k      : multiplier. k=1.5 with a normal-ish error distribution covers
             roughly 93% of cases on the safe side.
    """
    window: int = 48
    k: float = 1.5

    def __post_init__(self):
        self._errs: deque[float] = deque(maxlen=self.window)

    def update(self, y_true: float, y_pred: float) -> None:
        self._errs.append(float(y_pred - y_true))

    def dispersion(self) -> float:
        """Robust spread of recent errors.

        Uses the interquartile range scaled to a standard-deviation equivalent
        rather than a plain std, so one enormous burst error does not blow the
        margin up for the whole next window.
        """
        if len(self._errs) < 4:
            return 0.0
        a = np.fromiter(self._errs, dtype=float)
        iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
        return iqr / 1.349 if iqr > 0 else float(np.std(a))

    def margin(self) -> float:
        return self.k * self.dispersion()


def simulate_allocation(y_true: np.ndarray, y_pred: np.ndarray,
                        window: int = 48, k: float = 1.5) -> dict:
    """Replay a forecast series through the residual-margin allocator.

    Returns the operational metrics that actually matter - which is the point
    the project keeps making: a lower MAE is only interesting if it moves these.

      allocated        capacity committed each interval (forecast + margin)
      sla_violation    allocated < actual demand (the expensive failure)
      idle_capacity    allocated - actual, when positive (the wasteful failure)
      utilisation      served demand / total allocated, aggregated over the
                       whole run. Served demand is min(actual, allocated) - you
                       cannot consume more than was provisioned, so this can
                       never exceed 100%. Averaging the per-interval ratio
                       instead would let under-provisioned intervals report
                       >100% utilisation, which is meaningless.
    """
    tr = ResidualTracker(window=window, k=k)
    alloc, viol, idle = [], [], []

    for yt, yp in zip(np.asarray(y_true, float), np.asarray(y_pred, float)):
        a = yp + tr.margin()          # decide using ONLY what is known now
        alloc.append(a)
        viol.append(a < yt)
        idle.append(max(0.0, a - yt))
        tr.update(yt, yp)             # only now do we learn the truth

    alloc = np.asarray(alloc); idle = np.asarray(idle)
    y = np.asarray(y_true, float)

    served = np.minimum(y, alloc)          # you cannot use more than you got
    total_alloc = float(np.sum(np.clip(alloc, 0, None)))
    utilisation = (float(np.sum(served)) / total_alloc * 100) if total_alloc > 0 else float("nan")

    # Unmet demand: how much capacity was short, summed over violating intervals.
    # SLA violation RATE tells you how often; this tells you how badly.
    unmet = float(np.sum(np.clip(y - alloc, 0, None)))

    return {
        "sla_violation_pct": float(np.mean(viol) * 100),
        "mean_idle_capacity": float(np.mean(idle)),
        "mean_utilisation_pct": float(utilisation),
        "unmet_demand_total": unmet,
        "mean_allocated": float(np.mean(alloc)),
        "mean_margin": float(np.mean(alloc - np.asarray(y_pred, float))),
    }
