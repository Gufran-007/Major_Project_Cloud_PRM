#!/usr/bin/env python3
"""
Command-line entry point.

QUICK START (no data needed - uses the synthetic generator):

    python run.py --demo

RUN ON REAL BITBRAINS DATA:

    1. Download from http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains
    2. Unzip so you have a folder of per-VM .csv files
    3. Edit config/default.yaml:  use_synthetic: false
                                  data_dir: <path to fastStorage>
    4. python run.py --config config/default.yaml

OTHER USEFUL FLAGS:

    --ablation      also run the feature-group ablation
    --models a,b    override the model list for one run
"""
from __future__ import annotations

import argparse
import sys

from src.config import Config
from src import experiment


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="cloud-prm: predictive resource management for cloud data centres")
    ap.add_argument("--config", default=None, help="path to a YAML config file")
    ap.add_argument("--demo", action="store_true",
                    help="run on synthetic data with small defaults")
    ap.add_argument("--ablation", action="store_true",
                    help="also run the feature-group ablation study")
    ap.add_argument("--policy", action="store_true",
                    help="run the Phase-III predictive vs reactive comparison")
    ap.add_argument("--models", default=None,
                    help="comma-separated model list, overrides the config")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config)
    if args.demo:
        cfg.use_synthetic = True
        cfg.n_machines = min(cfg.n_machines, 6)
    if args.models:
        cfg.models = [m.strip() for m in args.models.split(",") if m.strip()]

    print("=" * 72)
    print(" cloud-prm  ·  Predictive Resource Management for Cloud Data Centres")
    print("=" * 72)

    res = experiment.run(cfg)

    print("\n" + "-" * 72)
    print(" SUMMARY  (pooled across machines, sorted best MAE first)")
    print("-" * 72)
    cols = ["model", "MAE_mean", "RMSE_mean", "under_rate_pct_mean",
            "sla_violation_pct_mean", "mean_utilisation_pct_mean"]
    cols = [c for c in cols if c in res["summary"].columns]
    print(res["summary"][cols].to_string(index=False))

    if not res["by_segment"].empty:
        print("\n" + "-" * 72)
        print(" BY PREDICTABILITY SEGMENT")
        print("-" * 72)
        print(res["by_segment"].to_string(index=False))

    if args.policy:
        print("\n" + "-" * 72)
        print(" PREDICTIVE vs REACTIVE ALLOCATION  (same trace, same delay)")
        print("-" * 72)
        pc = experiment.run_policy_comparison(cfg)
        if not pc.empty:
            print(pc.to_string(index=False))
            print("\n How to read this table:")
            print("  * Both policies were calibrated on VALIDATION to the same SLA")
            print("    violation target, so their violation rates should be close at")
            print("    each delay. Check that first - if they are far apart the")
            print("    comparison is between operating points, not policies.")
            print("  * Given matched service quality, the question is which policy")
            print("    wastes less: compare mean_utilisation_pct and mean_idle_capacity.")
            print("  * The provisioning delay is the mechanism by which prediction can")
            print("    help. Report the shape you actually observe across delays rather")
            print("    than assuming the advantage must grow.")

    if args.ablation:
        print("\n" + "-" * 72)
        print(" FEATURE-GROUP ABLATION  (higher MAE when dropped = more useful)")
        print("-" * 72)
        ab = experiment.run_ablation(cfg)
        if not ab.empty:
            print(ab.to_string(index=False))

    print("\nDone. CSVs are in", cfg.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
