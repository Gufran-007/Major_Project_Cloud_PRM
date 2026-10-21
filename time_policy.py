"""
time_policy.py - measures the two allocation-layer rows of Table 5.6.

Table 5.6 asks for the cost of the allocation policy itself, not just the
forecasters. Those two rows cannot come out of results_summary.csv, because the
experiment never times the policy separately. This does.

Run it from the folder that holds run.py:

    python time_policy.py

It prints the two numbers, and writes timing_policy.csv beside them.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import Config
from src import data as data_mod, features as feat_mod, models as models_mod
from src.splits import chronological_split, scale
from src.policy import PredictiveMarginPolicy
from src.simulator import replay

REPEATS = 5


def main():
    cfg = Config.load("config/default.yaml")
    machines = data_mod.load_dataset(cfg)
    name, mdf = next(iter(machines.items()))
    X, y, _ = feat_mod.build_features(mdf, cfg)
    raw = chronological_split(X, y, cfg)
    sp = scale(raw)

    key = cfg.policy_model if cfg.policy_model in models_mod.REGISTRY else "hgb"
    model = models_mod.build([key])[key]
    src = sp if model.needs_scaling else raw
    model.fit(src.X_train, src.y_train, src.X_val, src.y_val)
    fc = model.predict(src.X_test)
    demand = src.y_test.to_numpy()
    n = len(demand)

    # ---- one policy decision, in isolation -------------------------------
    # Timed over the whole test span and divided, because a single decide()
    # call is far below the clock's resolution.
    per_decision = []
    for _ in range(REPEATS):
        pol = PredictiveMarginPolicy(residual_window=cfg.residual_window,
                                     k=cfg.margin_k)
        t0 = time.perf_counter()
        for i in range(n):
            # the real per-interval cycle: decide, then learn the truth
            pol.decide(t=i, observed_now=float(demand[i]), forecast=float(fc[i]))
            pol.observe(float(demand[i]), float(fc[i]))
        per_decision.append((time.perf_counter() - t0) / n * 1000.0)

    # ---- a full replay of one machine ------------------------------------
    full = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        replay(demand, PredictiveMarginPolicy(residual_window=cfg.residual_window,
                                              k=cfg.margin_k),
               forecasts=fc, provisioning_delay=0)
        full.append((time.perf_counter() - t0) * 1000.0)

    dec = float(np.median(per_decision))
    rep = float(np.median(full))
    budget = 300_000.0

    print(f"\nmachine timed: {name}   test intervals: {n}   repeats: {REPEATS}")
    print("\nTable 5.6, the two allocation rows")
    print("-" * 64)
    print(f"Allocation policy (per decision)   training: n/a (no fitting)")
    print(f"                                   inference: {dec:.4f} ms per decision")
    print(f"                                   at 10,000 machines: "
          f"{dec * 10_000 / budget * 100:.2f}% of a 5-minute interval")
    print(f"Full replay of one machine         training: n/a")
    print(f"                                   wall clock: {rep:.1f} ms "
          f"for {n} intervals")
    print(f"                                   that is {rep / n:.4f} ms per interval")
    print("-" * 64)
    print("\nspread across repeats (ms): "
          f"per-decision {min(per_decision):.4f}-{max(per_decision):.4f}, "
          f"replay {min(full):.1f}-{max(full):.1f}")

    pd.DataFrame([
        {"component": "Allocation policy (per decision)", "training_s": None,
         "inference_ms": round(dec, 4), "n_intervals": n},
        {"component": "Full replay of one machine", "training_s": None,
         "inference_ms": round(rep, 1), "n_intervals": n},
    ]).to_csv("timing_policy.csv", index=False)
    print("\nwrote timing_policy.csv")


if __name__ == "__main__":
    main()
