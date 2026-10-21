"""
Feature construction: turn a time series into a supervised learning table.

THE CORE IDEA
-------------
A forecaster answers: "given everything I know at time t, what is demand at
t+h?" So every row of the output table must contain ONLY information that was
genuinely available at time t. The target column is shifted backwards by the
horizon to line the future value up with the present features.

Getting this wrong - letting any value from t+1 or later into the feature row
for t - is called leakage. It produces beautiful validation scores and a model
that is useless in production.

NOTE ON LAG NUMBERING (this trips people up):
    lag 0 = the value AT time t. This IS available when you make the decision,
            so it must be included - it is what "persistence" forecasts with.
    lag 1 = the value at t-1, and so on.
Excluding lag 0 would quietly turn a 1-step forecast into a 2-step one and
make every model look worse than it is.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import Config

RESOURCES = ["cpu_usage_pct", "mem_usage_pct", "disk_kbps", "net_kbps"]


def build_features(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Return (X, y, feature_group_labels) for one machine.

    feature_group_labels lets the ablation in experiment.py switch whole
    groups off without hard-coding column names anywhere else.
    """
    present = [c for c in RESOURCES if c in df.columns]
    X = pd.DataFrame(index=df.index)
    groups: List[str] = []

    def add(name: str, series: pd.Series, group: str):
        X[name] = series
        groups.append(group)

    # ---- GROUP 1: lagged utilisation --------------------------------------
    # The single most informative group. Lag 1 alone is the naive forecast.
    for col in present:
        for lag in cfg.lags:
            # shift(0) is a no-op and gives the current value - intentional.
            add(f"{col}_lag{lag}", df[col].shift(lag), "lag")

    # ---- GROUP 2: rolling statistics --------------------------------------
    # Level AND volatility. The std columns are what let a model behave
    # differently during calm and turbulent periods.
    # The window ENDS at t inclusive: the value at t is known at decision time.
    for col in present:
        for w in cfg.roll_windows:
            r = df[col].rolling(w, min_periods=max(2, w // 3))
            add(f"{col}_rmean{w}", r.mean(), "roll")
            add(f"{col}_rstd{w}",  r.std(),  "roll")
            add(f"{col}_rmax{w}",  r.max(),  "roll")

    # ---- GROUP 3: calendar --------------------------------------------------
    # Encoded as sine/cosine pairs rather than raw integers, so the model is not
    # told that hour 23 and hour 0 are 23 units apart. They are one hour apart.
    if cfg.use_calendar:
        hour = df.index.hour + df.index.minute / 60.0
        add("hour_sin", pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index), "cal")
        add("hour_cos", pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index), "cal")
        dow = df.index.dayofweek
        add("dow_sin", pd.Series(np.sin(2 * np.pi * dow / 7), index=df.index), "cal")
        add("dow_cos", pd.Series(np.cos(2 * np.pi * dow / 7), index=df.index), "cal")
        add("is_weekend", pd.Series((dow >= 5).astype(float), index=df.index), "cal")

    # ---- GROUP 4: cross-resource ratios -------------------------------------
    # The CPU:memory balance often identifies WHAT KIND of work a machine is
    # doing, which the absolute levels alone do not reveal.
    if cfg.use_cross_resource and {"cpu_usage_pct", "mem_usage_pct"} <= set(present):
        add("cpu_mem_ratio",
            (df["cpu_usage_pct"] / df["mem_usage_pct"].replace(0, np.nan)),
            "cross")
    if cfg.use_cross_resource and "disk_kbps" in present and "net_kbps" in present:
        add("io_total", (df["disk_kbps"] + df["net_kbps"]), "cross")

    # ---- target -------------------------------------------------------------
    # Shift NEGATIVE: pull the future value back to line up with today's row.
    y = df[cfg.target].shift(-cfg.horizon)

    # Drop rows that any shift left incomplete, and rows inside flagged gaps.
    valid = df["is_valid"] if "is_valid" in df else pd.Series(True, index=df.index)
    # A row is unusable if its TARGET falls inside a long gap, too.
    target_valid = valid.shift(-cfg.horizon).fillna(False).astype(bool)
    keep = X.notna().all(axis=1) & y.notna() & valid.astype(bool) & target_valid

    return X.loc[keep], y.loc[keep], groups


def select_groups(X: pd.DataFrame, groups: List[str], drop: str | None) -> pd.DataFrame:
    """Return X with one feature group removed - the ablation lever."""
    if drop is None:
        return X
    cols = [c for c, g in zip(X.columns, groups) if g != drop]
    return X[cols]
