"""
Loading, cleaning and resampling.

DESIGN NOTES YOU MAY BE ASKED ABOUT IN THE VIVA
-----------------------------------------------
* Short gaps are interpolated; LONG gaps are flagged and excluded.
  Filling a two-hour outage by interpolation invents a smooth straight line
  that no model can get wrong, which flatters every model equally and makes
  your comparison meaningless. The threshold is config.max_gap_intervals.

* Extreme values are KEPT, not clipped.
  A spike is the event the whole project exists to anticipate. Removing it as
  an "outlier" would delete the phenomenon under study.

* Normalisation is fitted on TRAINING DATA ONLY (see splits.py).
  Fitting a scaler on the full series leaks future information backwards and
  quietly inflates your scores. This is the single most common mistake in
  student time-series projects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from . import synthetic

# Bitbrains ships semicolon-separated files with these column headers.
# Mapping them to our own short names keeps the rest of the code readable
# and means switching datasets only touches this dictionary.
BITBRAINS_COLUMNS = {
    "Timestamp [ms]": "timestamp",
    "CPU usage [%]": "cpu_usage_pct",
    "Memory usage [KB]": "mem_usage_kb",
    "Memory capacity provisioned [KB]": "mem_prov_kb",
    "Disk read throughput [KB/s]": "disk_read_kbps",
    "Disk write throughput [KB/s]": "disk_write_kbps",
    "Network received throughput [KB/s]": "net_rx_kbps",
    "Network transmitted throughput [KB/s]": "net_tx_kbps",
}


def _load_bitbrains_file(path: Path) -> pd.DataFrame:
    """Read one Bitbrains per-VM CSV and normalise it to our column names."""
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=BITBRAINS_COLUMNS)

    # Bitbrains timestamps are UNIX seconds despite the "[ms]" header.
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    # Derive the percentage columns the rest of the pipeline expects.
    if "mem_usage_kb" in df and "mem_prov_kb" in df:
        prov = df["mem_prov_kb"].replace(0, np.nan)
        df["mem_usage_pct"] = (df["mem_usage_kb"] / prov * 100).clip(0, 100)
    df["disk_kbps"] = df.get("disk_read_kbps", 0) + df.get("disk_write_kbps", 0)
    df["net_kbps"] = df.get("net_rx_kbps", 0) + df.get("net_tx_kbps", 0)
    return df


def clean_and_resample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Put one machine onto a uniform grid and mark which rows are trustworthy.

    Returns the frame with an extra boolean column `is_valid`. Rows where
    is_valid is False sit inside a long gap: they are kept so the time axis
    stays continuous, but they are dropped before training and before scoring.
    """
    out = (df.set_index("timestamp")
             .sort_index()
             .resample(cfg.resample_rule)
             .mean(numeric_only=True))

    target = cfg.target
    missing = out[target].isna()

    # Label each run of consecutive NaNs so we can measure how long it is.
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    long_gap = missing & (run_len > cfg.max_gap_intervals)

    # Short gaps: interpolate. Long gaps: fill so arithmetic works, but flag.
    out = out.interpolate(method="time", limit_direction="both")
    out["is_valid"] = ~long_gap

    n_bad = int((~out["is_valid"]).sum())
    if n_bad:
        print(f"    flagged {n_bad} intervals inside long gaps (excluded from scoring)")
    return out


def load_dataset(cfg: Config) -> Dict[str, pd.DataFrame]:
    """Load every machine, cleaned and resampled, keyed by machine id."""
    if cfg.use_synthetic:
        print(f"[data] generating {cfg.n_machines} synthetic machines "
              f"(set use_synthetic: false in the config to use real traces)")
        raw = synthetic.make_dataset(cfg.n_machines, seed=cfg.seed)
    else:
        folder = Path(cfg.data_dir)
        if not folder.exists():
            raise FileNotFoundError(
                f"{folder} not found. Download the Bitbrains traces from "
                "http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains and point "
                "data_dir at the fastStorage folder, or set use_synthetic: true."
            )
        files = sorted(folder.glob("*.csv"))[: cfg.n_machines]
        if not files:
            raise FileNotFoundError(f"No .csv files inside {folder}")
        print(f"[data] loading {len(files)} Bitbrains traces from {folder}")
        raw = {f.stem: _load_bitbrains_file(f) for f in files}

    cleaned = {}
    for name, df in raw.items():
        print(f"  - {name}")
        cleaned[name] = clean_and_resample(df, cfg)
    return cleaned
