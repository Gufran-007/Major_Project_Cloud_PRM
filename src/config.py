"""
Experiment configuration.

WHY THIS FILE EXISTS
--------------------
Phase-I promised that every reported figure would be reproducible from a
configuration file plus a fixed random seed.  That promise only holds if the
settings live in data, not scattered through the code as literals.  So every
tunable number in this project comes from here, and an experiment is fully
described by one YAML file that you can commit alongside your results.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class Config:
    # ---- data ---------------------------------------------------------
    data_dir: str = "data/bitbrains/fastStorage"   # folder of per-VM CSV files
    use_synthetic: bool = True        # True = generate fake data, ignore data_dir
    n_machines: int = 12              # how many VMs to load (keep small while testing)
    resample_rule: str = "5min"       # the uniform grid every series is put onto
    max_gap_intervals: int = 6        # gaps longer than this are FLAGGED, not filled
    target: str = "cpu_usage_pct"     # the column we are forecasting

    # ---- features -----------------------------------------------------
    lags: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 6, 12, 24])
    roll_windows: List[int] = field(default_factory=lambda: [6, 12, 48])
    use_calendar: bool = True
    use_cross_resource: bool = True
    horizon: int = 1                  # how many intervals ahead to predict

    # ---- splitting ----------------------------------------------------
    train_frac: float = 0.60
    val_frac: float = 0.20            # test gets whatever is left

    # ---- models -------------------------------------------------------
    models: List[str] = field(default_factory=lambda: [
        "naive", "seasonal_naive", "ridge", "hgb",
    ])

    # ---- allocation ---------------------------------------------------
    residual_window: int = 48         # intervals used to size the safety margin
    margin_k: float = 1.5             # margin = k * dispersion of recent residuals
    policy_model: str = "hgb"         # forecaster that drives the predictive policy
    provisioning_delays: List[int] = field(default_factory=lambda: [0, 1, 3, 6])
    cost_alpha: float = 1.0           # price of one unit of idle capacity
    cost_beta: float = 5.0            # price of one SLA violation
    target_violation_pct: float = 5.0  # both policies calibrated to this on validation

    # ---- bookkeeping --------------------------------------------------
    seed: int = 42
    output_dir: str = "outputs"

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Load a YAML config. Missing file -> all defaults (handy for a demo run)."""
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            # Fail loudly. A silently ignored typo in a config is the kind of bug
            # that makes two "identical" experiments quietly differ.
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
