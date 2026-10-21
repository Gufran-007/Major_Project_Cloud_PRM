"""
Experiment orchestration - the file that produces your results tables.

WHAT IT DOES, IN ORDER
----------------------
 1. load + clean every machine                       (data.py)
 2. build the supervised feature table per machine    (features.py)
 3. split chronologically, scale on train only        (splits.py)
 4. fit every model behind the common interface       (models.py)
 5. score them, split by over/under-prediction        (metrics.py)
 6. replay the forecasts through the margin allocator (residuals.py)
 7. write CSVs you can paste straight into the report

The per-machine loop matters: Phase-II Section 2.6 committed to reporting per
machine SEGMENT rather than pooled, because machines differ enormously in
predictability and a single pooled average hides that completely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from . import data as data_mod
from . import features as feat_mod
from . import models as models_mod
from . import metrics as metrics_mod
from . import residuals as resid_mod
from .splits import chronological_split, scale
from . import simulator as sim_mod


def run(cfg: Config) -> Dict[str, pd.DataFrame]:
    np.random.seed(cfg.seed)
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    machines = data_mod.load_dataset(cfg)
    rows: List[dict] = []

    print(f"\n[experiment] models requested: {cfg.models}")
    for mname, mdf in machines.items():
        X, y, groups = feat_mod.build_features(mdf, cfg)
        if len(X) < 200:
            print(f"[skip] {mname}: only {len(X)} usable rows after cleaning")
            continue

        raw_split = chronological_split(X, y, cfg)
        scaled_split = scale(raw_split)
        print(f"\n[{mname}] {raw_split.sizes()}  features={X.shape[1]}")

        built = models_mod.build(cfg.models)
        naive_mae = None

        for key, model in built.items():
            sp = scaled_split if model.needs_scaling else raw_split
            model.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val)
            pred = model.predict(sp.X_test)

            m = metrics_mod.evaluate(sp.y_test, pred)
            if key == "naive":
                naive_mae = m["MAE"]

            # Operational outcome, not just forecast error. This is the number
            # the project actually cares about.
            alloc = resid_mod.simulate_allocation(
                sp.y_test.to_numpy(), pred,
                window=cfg.residual_window, k=cfg.margin_k)

            rows.append({
                "machine": mname, "model": key,
                **{k: round(v, 4) for k, v in m.items()},
                **{k: round(v, 4) for k, v in alloc.items()},
                "fit_seconds": round(model.fit_seconds, 3),
                "predict_ms_per_row": round(model.predict_ms_per_row, 4),
            })
            print(f"    {key:16s} MAE={m['MAE']:6.3f}  RMSE={m['RMSE']:6.3f}  "
                  f"under={m['under_rate_pct']:5.1f}%  "
                  f"SLA_viol={alloc['sla_violation_pct']:5.1f}%  "
                  f"util={alloc['mean_utilisation_pct']:5.1f}%")

        # skill score needs the naive MAE, so add it after the loop
        if naive_mae:
            for r in rows:
                if r["machine"] == mname and "skill_vs_naive_pct" not in r:
                    r["skill_vs_naive_pct"] = round(
                        metrics_mod.skill_vs_naive(r["MAE"], naive_mae), 2)

    per_machine = pd.DataFrame(rows)
    if per_machine.empty:
        raise RuntimeError("No results produced - check the data path and config.")

    # ---- pooled summary across machines (mean +/- std) --------------------
    num = per_machine.select_dtypes("number").columns
    summary = (per_machine.groupby("model")[list(num)]
               .agg(["mean", "std"]).round(4))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index().sort_values("MAE_mean")

    # ---- predictability segments -----------------------------------------
    # Split machines by how well the naive baseline does on them. "Hard"
    # machines are where a learned model has room to help - and where it may
    # fail. Reporting the split is more informative than one average.
    nv = per_machine[per_machine["model"] == "naive"][["machine", "MAE"]]
    if not nv.empty:
        cut = nv["MAE"].median()
        seg = {r.machine: ("predictable" if r.MAE <= cut else "volatile")
               for r in nv.itertuples()}
        per_machine["segment"] = per_machine["machine"].map(seg)
        by_segment = (per_machine.groupby(["segment", "model"])
                      [["MAE", "RMSE", "under_rate_pct",
                        "sla_violation_pct", "mean_utilisation_pct"]]
                      .mean().round(4).reset_index())
    else:
        by_segment = pd.DataFrame()

    per_machine.to_csv(out_dir / "results_per_machine.csv", index=False)
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    if not by_segment.empty:
        by_segment.to_csv(out_dir / "results_by_segment.csv", index=False)
    pd.Series(cfg.as_dict()).to_json(out_dir / "config_used.json", indent=2)

    print(f"\n[experiment] wrote CSVs to {out_dir.resolve()}")
    return {"per_machine": per_machine, "summary": summary, "by_segment": by_segment}


def run_policy_comparison(cfg: Config) -> pd.DataFrame:
    """PHASE-III MEASUREMENT: predictive vs reactive allocation.

    For each machine we take the best available forecaster, produce forecasts
    over the test span, then replay that span twice - once through the reactive
    threshold controller and once through the predictive margin policy - at
    several provisioning delays.

    Both policies see the SAME trace and are charged the SAME delay. That is
    what makes the difference attributable to the policy rather than to the
    setup.
    """
    machines = data_mod.load_dataset(cfg)
    rows = []
    for mname, mdf in machines.items():
        X, y, groups = feat_mod.build_features(mdf, cfg)
        if len(X) < 300:
            continue
        raw = chronological_split(X, y, cfg)
        scaled = scale(raw)

        # Pick the forecaster named in the config; fall back to gradient boosting.
        key = cfg.policy_model if cfg.policy_model in models_mod.REGISTRY else "hgb"
        built = models_mod.build([key])
        if not built:
            print(f"[policy] forecaster '{key}' unavailable - skipping {mname}")
            continue
        model = built[key]
        sp = scaled if model.needs_scaling else raw
        model.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val)
        fc_val = model.predict(sp.X_val)
        fc_test = model.predict(sp.X_test)
        demand_val = sp.y_val.to_numpy()
        demand_test = sp.y_test.to_numpy()

        # Calibrate BOTH policies on validation to the same SLA target, then
        # evaluate on test. Without this the comparison is between two
        # arbitrary points on a trade-off curve rather than between policies.
        for r in sim_mod.compare_calibrated(
                demand_val, fc_val, demand_test, fc_test,
                delays=tuple(cfg.provisioning_delays),
                target_violation_pct=cfg.target_violation_pct,
                residual_window=cfg.residual_window):
            rows.append({"machine": mname, "forecaster": key, **r})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    agg = (df.groupby(["provisioning_delay", "policy"])
           [["sla_violation_pct", "mean_utilisation_pct", "mean_idle_capacity",
             "unmet_demand_total", "energy_proxy_wh", "cost_J"]]
           .mean().round(4).reset_index())

    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results_policy_per_machine.csv", index=False)
    agg.to_csv(out / "results_policy_comparison.csv", index=False)
    print(f"\n[policy] wrote {out / 'results_policy_comparison.csv'}")
    return agg


def run_ablation(cfg: Config) -> pd.DataFrame:
    """Drop each feature group in turn and see what the model loses.

    This answers 'do the calendar features actually earn their place?' with
    evidence instead of assumption. Uses one model (hgb) to keep it cheap.
    """
    machines = data_mod.load_dataset(cfg)
    rows = []
    for mname, mdf in machines.items():
        X, y, groups = feat_mod.build_features(mdf, cfg)
        if len(X) < 200:
            continue
        for drop in [None, "lag", "roll", "cal", "cross"]:
            Xs = feat_mod.select_groups(X, groups, drop)
            if Xs.shape[1] == 0:
                continue
            sp = chronological_split(Xs, y, cfg)
            mdl = models_mod.HistGBForecaster().fit(sp.X_train, sp.y_train,
                                                    sp.X_val, sp.y_val)
            m = metrics_mod.evaluate(sp.y_test, mdl.predict(sp.X_test))
            rows.append({"machine": mname,
                         "dropped_group": drop or "none (full set)",
                         "n_features": Xs.shape[1],
                         "MAE": round(m["MAE"], 4),
                         "RMSE": round(m["RMSE"], 4)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (df.groupby("dropped_group")[["MAE", "RMSE", "n_features"]]
           .mean().round(4).reset_index().sort_values("MAE"))
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out / "results_ablation.csv", index=False)
    print(f"\n[ablation] wrote {out / 'results_ablation.csv'}")
    return agg
