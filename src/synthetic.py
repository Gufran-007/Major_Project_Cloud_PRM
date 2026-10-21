"""
Synthetic trace generator.

WHY THIS FILE EXISTS
--------------------
Two reasons, and the second matters more than the first.

1. It lets anyone run the whole pipeline immediately, before downloading 
   several gigabytes of Bitbrains traces. Useful for a demo.

2. It gives you data whose *correct answer you already know*. The generator
   builds each series from an explicit daily cycle + weekly cycle + noise. So
   if your pipeline is working, a decent model should clearly beat the naive
   baseline on synthetic data. If it does NOT, the bug is in your code, not in
   the real world. This is the cheapest debugging tool in the project - use it
   whenever a result on real data looks suspicious.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_machine(n_intervals: int, rng: np.random.Generator,
                 freq_minutes: int = 5, start: str = "2013-08-01") -> pd.DataFrame:
    """Generate one VM's trace with realistic-looking structure.

    The shape is deliberate:
      - a daily sinusoid (business hours busier than nights)
      - a weekly term (weekends quieter)
      - an AR(1) drift so consecutive samples are correlated, as real ones are
      - occasional bursts, because bursts are the thing we actually care about
      - a few genuine gaps, so the cleaning code in data.py has something to do
    """
    idx = pd.date_range(start, periods=n_intervals, freq=f"{freq_minutes}min")
    t = np.arange(n_intervals)

    per_day = 24 * 60 // freq_minutes
    daily = 18 * np.sin(2 * np.pi * t / per_day - np.pi / 2)
    weekly = 8 * np.sin(2 * np.pi * t / (per_day * 7))

    # AR(1) drift: today's level depends on yesterday's. phi<1 keeps it stable.
    phi, drift = 0.97, np.zeros(n_intervals)
    innov = rng.normal(0, 1.2, n_intervals)
    for i in range(1, n_intervals):
        drift[i] = phi * drift[i - 1] + innov[i]

    base = float(rng.uniform(18, 45))
    cpu = base + daily + weekly + drift + rng.normal(0, 1.5, n_intervals)

    # Bursts: short, sharp, and the reason predictive allocation exists at all.
    for _ in range(max(1, n_intervals // 900)):
        s = int(rng.integers(0, n_intervals - 24))
        dur = int(rng.integers(3, 18))
        cpu[s:s + dur] += rng.uniform(15, 45)

    cpu = np.clip(cpu, 0, 100)

    # Memory correlates with CPU but is smoother and slower to react.
    mem = np.clip(0.55 * cpu + 20 + np.convolve(
        rng.normal(0, 2, n_intervals), np.ones(12) / 12, mode="same"), 0, 100)

    df = pd.DataFrame({
        "timestamp": idx,
        "cpu_usage_pct": cpu,
        "mem_usage_pct": mem,
        "disk_kbps": np.abs(rng.normal(120, 60, n_intervals)) + cpu * 2,
        "net_kbps": np.abs(rng.normal(90, 40, n_intervals)) + cpu * 1.4,
    })

    # Punch a few holes so the gap-handling path is genuinely exercised.
    for _ in range(3):
        s = int(rng.integers(0, n_intervals - 40))
        df.loc[s:s + int(rng.integers(2, 30)),
               ["cpu_usage_pct", "mem_usage_pct", "disk_kbps", "net_kbps"]] = np.nan
    return df


def make_dataset(n_machines: int, n_intervals: int = 4032, seed: int = 42):
    """n_intervals=4032 is exactly 14 days at a 5-minute interval."""
    rng = np.random.default_rng(seed)
    return {f"synthetic_vm_{i+1:03d}": make_machine(n_intervals, rng)
            for i in range(n_machines)}
