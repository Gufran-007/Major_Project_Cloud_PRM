"""
Discrete-event trace replay.

WHAT THIS SIMULATES, AND WHAT IT DOES NOT
-----------------------------------------
It replays a recorded demand trace against an allocation policy and charges a
PROVISIONING DELAY: a decision taken at interval t does not take effect until
t + delay. Booting a VM, warming a cache or migrating state takes time, and
during that interval the workload is served by whatever capacity was already
in place.

That delay is the entire mechanism by which forecasting can win. With delay=0 a
reactive controller sees the truth and acts on it instantly, and prediction has
nothing to offer. As the delay grows, acting on what has already happened gets
progressively worse. If your results show no advantage at delay=0 and a growing
advantage as delay increases, that is the expected shape - and if they do not,
that is a finding worth reporting rather than hiding.

NOT simulated: VM migration, network topology, storage placement, multi-tenant
interference, or real power draw. Those were explicitly out of scope in
Phase-I. Energy is a utilisation-derived proxy, and the report says so wherever
it is quoted.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .policy import BasePolicy, asymmetric_cost


@dataclass
class SimResult:
    policy: str
    demand: np.ndarray
    allocated: np.ndarray
    margin: np.ndarray
    metrics: Dict[str, float] = field(default_factory=dict)


def replay(demand: np.ndarray,
           policy: BasePolicy,
           forecasts: Optional[np.ndarray] = None,
           provisioning_delay: int = 3,
           idle_power_w: float = 120.0,
           peak_power_w: float = 300.0) -> SimResult:
    """Run one policy over one demand trace.

    Parameters
    ----------
    demand              the realised resource demand, per interval
    policy              anything implementing BasePolicy
    forecasts           predicted demand for interval t+delay, issued at t.
                        None for policies that do not use a forecast.
    provisioning_delay  intervals between deciding and the capacity being live

    The timing is the part to get right, so it is spelled out:

        at interval t   the policy sees demand[t-1] - the last interval that
                        has actually FINISHED - and, if it has one, a forecast
                        of demand[t + delay]. It never sees demand[t], because
                        interval t is still in progress when the decision is
                        made. Letting it peek at demand[t] would make a
                        zero-delay policy trivially perfect and the whole
                        comparison meaningless.
                        it commits a capacity value
        at interval t+delay   that capacity becomes effective
        serving         demand[t] is served by whatever was committed at
                        t - delay

    A policy is therefore never scored against a decision it could not have made.
    """
    n = len(demand)
    demand = np.asarray(demand, dtype=float)
    delay = int(provisioning_delay)

    # Capacity decided but not yet live. Length == delay, seeded with the first
    # observed level so the opening intervals are not scored against an empty
    # pipeline. delay == 0 means a decision takes effect within the same
    # interval, which is the idealised no-latency case.
    pipeline: deque[float] = deque([float(demand[0])] * delay, maxlen=delay) \
        if delay > 0 else deque()

    allocated = np.zeros(n)
    margins = np.zeros(n)

    for t in range(n):
        fc = None
        if forecasts is not None and t < len(forecasts):
            fc = float(forecasts[t])

        if delay == 0:
            # decide, and it is live immediately
            last_finished = float(demand[t - 1]) if t > 0 else float(demand[0])
            d = policy.decide(t, observed_now=last_finished, forecast=fc)
            margins[t] = d.margin
            allocated[t] = d.capacity
        else:
            # whatever was decided `delay` intervals ago is live NOW ...
            allocated[t] = pipeline[0]
            # ... and today's decision lands at t + delay
            last_finished = float(demand[t - 1]) if t > 0 else float(demand[0])
            d = policy.decide(t, observed_now=last_finished, forecast=fc)
            margins[t] = d.margin
            pipeline.append(d.capacity)

        # the truth for THIS interval is now known - feed it back
        policy.observe(float(demand[t]), fc)

    served = np.minimum(demand, allocated)
    idle = np.clip(allocated - demand, 0, None)
    unmet = np.clip(demand - allocated, 0, None)
    total_alloc = float(np.sum(np.clip(allocated, 0, None)))

    # Energy proxy: the standard linear utilisation-to-power model from the
    # consolidation literature, scaled by HOW MUCH capacity is provisioned.
    #
    # Scaling matters. An earlier version computed power from the utilisation
    # ratio alone, which rewarded a policy for provisioning a large idle pool -
    # exactly backwards. Capacity here is a percentage of one machine, so
    # allocated/100 is the number of machine-equivalents kept switched on; each
    # draws idle power just for being on, plus a proportional amount for the
    # work it actually does. A proxy, never presented as a measurement.
    util_frac = np.clip(served / np.where(allocated <= 0, np.nan, allocated), 0, 1)
    machines_on = np.clip(allocated, 0, None) / 100.0
    power = machines_on * (idle_power_w +
                           (peak_power_w - idle_power_w) * np.nan_to_num(util_frac))
    energy_wh = float(np.nansum(power) * (5.0 / 60.0))    # 5-minute intervals

    cost = asymmetric_cost(demand, allocated)

    metrics = {
        "sla_violation_pct": float(np.mean(allocated < demand) * 100),
        "mean_utilisation_pct": (float(np.sum(served)) / total_alloc * 100)
                                if total_alloc > 0 else float("nan"),
        "mean_idle_capacity": float(np.mean(idle)),
        "unmet_demand_total": float(np.sum(unmet)),
        "mean_allocated": float(np.mean(allocated)),
        "mean_margin": float(np.mean(margins)),
        "energy_proxy_wh": energy_wh,
        "cost_J": cost["cost_J"],
        "provisioning_delay": provisioning_delay,
    }
    return SimResult(policy=policy.name, demand=demand,
                     allocated=allocated, margin=margins, metrics=metrics)


def compare_policies(demand: np.ndarray, forecasts: np.ndarray,
                     delays: List[int] = (0, 1, 3, 6),
                     residual_window: int = 48, k: float = 1.5) -> List[dict]:
    """Run predictive and reactive over the SAME trace at several delays.

    Returns one row per (delay, policy). Sweeping the delay is what turns a
    single number into an argument: it shows the CONDITIONS under which
    prediction helps, instead of asserting that it always does.
    """
    from .policy import ReactiveThresholdPolicy, PredictiveMarginPolicy

    rows = []
    for delay in delays:
        for make in (
            lambda: ReactiveThresholdPolicy(initial=float(np.mean(demand[:20]))),
            lambda: PredictiveMarginPolicy(residual_window=residual_window, k=k),
        ):
            pol = make()
            fc = forecasts if pol.name == "predictive" else None
            res = replay(demand, pol, forecasts=fc, provisioning_delay=delay)
            rows.append({"provisioning_delay": delay, "policy": pol.name,
                         **{kk: round(vv, 4) for kk, vv in res.metrics.items()}})
    return rows


# ======================================================================
# Calibration: compare policies AT THE SAME OPERATING POINT
# ======================================================================

def _violation_rate(demand, policy_factory, forecasts, delay) -> float:
    return replay(demand, policy_factory(), forecasts=forecasts,
                  provisioning_delay=delay).metrics["sla_violation_pct"]


def calibrate(demand: np.ndarray, forecasts: Optional[np.ndarray],
              policy_factory, param_grid: List[float],
              target_violation_pct: float, delay: int) -> float:
    """Pick the parameter whose violation rate is closest to the target.

    WHY THIS EXISTS - and it is a methodological point worth defending
    ------------------------------------------------------------------
    A conservative policy can always buy a lower SLA violation rate by
    over-provisioning, and an aggressive one can always buy higher utilisation
    by accepting more breaches. Comparing an arbitrary reactive configuration
    against an arbitrary predictive one therefore compares two different points
    on a trade-off curve, not two policies - and whichever you happened to tune
    harder appears to win.

    So both policies are first calibrated on the VALIDATION span to hit the same
    SLA violation target, and only then compared on the test span. After
    calibration the question becomes the right one: at equal service quality,
    which policy wastes less capacity?
    """
    best, best_gap = param_grid[0], float("inf")
    for p in param_grid:
        v = _violation_rate(demand, lambda: policy_factory(p), forecasts, delay)
        gap = abs(v - target_violation_pct)
        if gap < best_gap:
            best, best_gap = p, gap
    return best


def compare_calibrated(demand_val: np.ndarray, fc_val: Optional[np.ndarray],
                       demand_test: np.ndarray, fc_test: Optional[np.ndarray],
                       delays=(0, 1, 3, 6),
                       target_violation_pct: float = 5.0,
                       residual_window: int = 48) -> List[dict]:
    """Calibrate on validation, evaluate on test, at several delays.

    Returns one row per (delay, policy) with the calibrated parameter recorded,
    so the reader can see what was tuned rather than having to trust that it was
    done fairly.
    """
    from .policy import ReactiveThresholdPolicy, PredictiveMarginPolicy

    rows = []
    for delay in delays:
        # --- predictive: tune the margin multiplier k --------------------
        k_grid = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        k = calibrate(demand_val, fc_val,
                      lambda p: PredictiveMarginPolicy(residual_window=residual_window, k=p),
                      k_grid, target_violation_pct, delay)
        res_p = replay(demand_test, PredictiveMarginPolicy(residual_window=residual_window, k=k),
                       forecasts=fc_test, provisioning_delay=delay)
        rows.append({"provisioning_delay": delay, "policy": "predictive",
                     "calibrated_param": f"k={k}",
                     **{kk: round(vv, 4) for kk, vv in res_p.metrics.items()}})

        # --- reactive: tune the scale-up threshold ------------------------
        up_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        init = float(np.mean(demand_val[:20])) if len(demand_val) >= 20 else float(np.mean(demand_val))
        up = calibrate(demand_val, None,
                       lambda p: ReactiveThresholdPolicy(scale_up_at=p, initial=init),
                       up_grid, target_violation_pct, delay)
        init_t = float(np.mean(demand_test[:20]))
        res_r = replay(demand_test, ReactiveThresholdPolicy(scale_up_at=up, initial=init_t),
                       forecasts=None, provisioning_delay=delay)
        rows.append({"provisioning_delay": delay, "policy": "reactive",
                     "calibrated_param": f"scale_up_at={up}",
                     **{kk: round(vv, 4) for kk, vv in res_r.metrics.items()}})
    return rows
