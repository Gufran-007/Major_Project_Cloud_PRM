"""
Allocation policies: the thing that turns a forecast into a capacity commitment.

WHY BOTH POLICIES LIVE IN ONE FILE
----------------------------------
The project's central measurement is predictive allocation versus reactive
allocation. For that comparison to mean anything, both must be driven by the
same simulation loop, charged the same provisioning delay, and scored by the
same metrics. Keeping them side by side makes it obvious that neither is being
given an advantage the other does not get.

THE ASYMMETRY THAT DRIVES EVERYTHING
------------------------------------
    over-provision  -> idle capacity      (costs money and power)
    under-provision -> unmet demand       (breaches the SLA)

These are not equally bad, so the cost function is deliberately asymmetric:

    J = alpha * E[(allocated - demand)+]  +  beta * P(allocated < demand)

alpha and beta are configurable because the trade-off is a commercial judgement
belonging to the operator, not a constant we get to pick for them. Reporting a
curve across several beta/alpha ratios is more honest than reporting one point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .residuals import ResidualTracker


@dataclass
class PolicyDecision:
    """What a policy returns each interval. Deliberately minimal."""
    capacity: float
    margin: float
    reason: str = ""


class BasePolicy:
    """Common interface. The simulator only ever calls decide()."""
    name = "base"

    def decide(self, t: int, observed_now: float,
               forecast: Optional[float]) -> PolicyDecision:
        raise NotImplementedError

    def observe(self, y_true: float, y_pred: Optional[float]) -> None:
        """Called after the truth is known. Default: ignore it."""
        return None


# ======================================================================
# BASELINE: reactive threshold scaling  (what production autoscalers do)
# ======================================================================

class ReactiveThresholdPolicy(BasePolicy):
    """Scale up when recent utilisation crosses a high-water mark, down on a low.

    This is the honest baseline. It is NOT a straw man: the thresholds and step
    size get the same tuning treatment as the predictive policy's parameters,
    because a comparison against a badly configured baseline proves nothing.

    Its structural weakness is not the thresholds - it is that the decision is
    made from what has ALREADY happened. Combined with a provisioning delay,
    that is the gap predictive allocation is supposed to close. If our results
    show no gap, the honest conclusion is that prediction did not help here.
    """
    name = "reactive"

    def __init__(self, scale_up_at: float = 0.80, scale_down_at: float = 0.50,
                 step: float = 0.20, initial: float = 50.0,
                 min_capacity: float = 1.0, smoothing: int = 3):
        self.up, self.down, self.step = scale_up_at, scale_down_at, step
        self.capacity = float(initial)
        self.min_capacity = min_capacity
        self.smoothing = smoothing
        self._recent: list[float] = []

    def decide(self, t: int, observed_now: float, forecast=None) -> PolicyDecision:
        self._recent.append(observed_now)
        if len(self._recent) > self.smoothing:
            self._recent.pop(0)
        # Smoothed so a single spike does not cause a scale-up-then-down flap.
        level = float(np.mean(self._recent))
        util = level / max(self.capacity, 1e-9)

        reason = "hold"
        if util > self.up:
            self.capacity *= (1.0 + self.step)
            reason = "scale up"
        elif util < self.down:
            self.capacity = max(self.min_capacity, self.capacity * (1.0 - self.step))
            reason = "scale down"

        return PolicyDecision(capacity=self.capacity, margin=0.0, reason=reason)


# ======================================================================
# PREDICTIVE: forecast + rolling-residual margin
# ======================================================================

class PredictiveMarginPolicy(BasePolicy):
    """Commit to the forecast plus a margin sized from the model's recent errors.

    The margin is NOT a constant. It comes from residuals.py, so it narrows when
    the forecaster has been reliable and widens when it has not. That is the
    Phase-I Section 5.3 mechanism, and it is what stops this policy from being
    "reactive scaling with extra steps".

    Note this policy never looks at the model itself - only at the forecast
    value and its own error history. That restriction is what keeps the
    swap-the-forecaster comparison valid.
    """
    name = "predictive"

    def __init__(self, residual_window: int = 48, k: float = 1.5,
                 min_capacity: float = 1.0, floor_on_observed: bool = True):
        self.tracker = ResidualTracker(window=residual_window, k=k)
        self.min_capacity = min_capacity
        # Never commit below what we can already see being used. Without this a
        # badly wrong low forecast guarantees a violation the policy could have
        # avoided using information it already had.
        self.floor_on_observed = floor_on_observed

    def decide(self, t: int, observed_now: float,
               forecast: Optional[float]) -> PolicyDecision:
        if forecast is None:                      # no forecast yet: fall back
            return PolicyDecision(capacity=max(self.min_capacity, observed_now),
                                  margin=0.0, reason="warmup")
        margin = self.tracker.margin()
        cap = forecast + margin
        if self.floor_on_observed:
            cap = max(cap, observed_now)
        return PolicyDecision(capacity=max(self.min_capacity, cap),
                              margin=margin, reason="forecast+margin")

    def observe(self, y_true: float, y_pred: Optional[float]) -> None:
        if y_pred is not None:
            self.tracker.update(y_true, y_pred)


# ======================================================================
# The asymmetric cost function from the design
# ======================================================================

def asymmetric_cost(demand: np.ndarray, allocated: np.ndarray,
                    alpha: float = 1.0, beta: float = 5.0,
                    gamma: float = 0.0) -> dict:
    """J = alpha * mean(idle)  +  beta * P(violation)  +  gamma * mean(unmet)

    The first two terms are the cost function specified in Phase-I. beta
    defaults to 5x alpha, i.e. "one SLA breach costs about as much as five units
    of wasted capacity". That ratio stands in for a commercial judgement - sweep
    it and report the curve rather than defending a single value.

    THE gamma TERM AND WHY IT WAS ADDED (Phase-III)
    -----------------------------------------------
    A unit test exposed a real gap in the specified form. Because the second
    term counts the FREQUENCY of violations and not their SIZE, being one unit
    short is charged exactly as much as being fifty units short. Two policies
    with identical violation rates and wildly different severity therefore score
    the same, which is not what any operator means by cost.

    gamma prices the average shortfall and closes that gap. It defaults to 0.0
    so that the Phase-I specification is reproduced exactly unless it is
    switched on, and results reported under either form say which was used.
    Real SLAs usually penalise both frequency and severity, so gamma > 0 is the
    more realistic setting.
    """
    d, a = np.asarray(demand, float), np.asarray(allocated, float)
    idle = np.clip(a - d, 0, None)
    unmet = np.clip(d - a, 0, None)
    viol = (a < d)
    return {
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "mean_idle": float(np.mean(idle)),
        "violation_rate": float(np.mean(viol)),
        "mean_unmet": float(np.mean(unmet)),
        "cost_J": float(alpha * np.mean(idle) + beta * np.mean(viol)
                        + gamma * np.mean(unmet)),
    }
