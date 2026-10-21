"""
fill_tables.py - reads the CSVs the experiment already wrote and prints the
Phase-III report tables with the empty cells filled in.

Usage (from the folder that holds run.py):

    python fill_tables.py
    python fill_tables.py --root .          # same thing
    python fill_tables.py --root C:\\path\\to\\cloud-prm

It looks for every directory under --root that contains result CSVs, works out
which one is the default run and which ones are the residual-window sweep, and
prints Tables 4 to 9. Nothing is invented: a cell with no data behind it stays
as [no data] so you can see what is still missing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ALPHA_DEFAULT = 1.0
BETA_DEFAULT = 5.0
TARGET_DEFAULT = 5.0

PRETTY_MODEL = {
    "naive": "Naive persistence",
    "seasonal_naive": "Seasonal naive",
    "arima": "ARIMA",
    "ridge": "Ridge regression",
    "hgb": "Gradient boosting",
    "lstm": "LSTM",
    "gru": "GRU",
}


# ---------------------------------------------------------------- discovery

def find_result_dirs(root: Path) -> list[Path]:
    """Any directory holding at least one results_*.csv."""
    hits = set()
    for p in root.rglob("results_*.csv"):
        hits.add(p.parent)
    return sorted(hits)


def read_window(d: Path) -> int | None:
    """Residual window for a run: from config_used.json, else from the folder name."""
    cfg = d / "config_used.json"
    if cfg.exists():
        try:
            raw = json.loads(cfg.read_text())
            if "residual_window" in raw:
                return int(raw["residual_window"])
        except Exception:
            pass
    m = re.search(r"w(\d+)", d.name)
    return int(m.group(1)) if m else None


def read_cost_weights(d: Path) -> tuple[float, float]:
    cfg = d / "config_used.json"
    if cfg.exists():
        try:
            raw = json.loads(cfg.read_text())
            return (float(raw.get("cost_alpha", ALPHA_DEFAULT)),
                    float(raw.get("cost_beta", BETA_DEFAULT)))
        except Exception:
            pass
    return ALPHA_DEFAULT, BETA_DEFAULT


def load(d: Path, name: str) -> pd.DataFrame | None:
    f = d / name
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
        return df if not df.empty else None
    except Exception:
        return None


# ---------------------------------------------------------------- helpers

def fmt(v, nd=2):
    if v is None or pd.isna(v):
        return "[no data]"
    return f"{v:.{nd}f}"


def table(title, header, rows, note=None):
    print()
    print(f"### {title}")
    if note:
        print(f"_{note}_")
    print()
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def mode_and_range(series: pd.Series) -> str:
    """Most frequently selected parameter value, with the spread if it varied."""
    vals = series.dropna().astype(str)
    if vals.empty:
        return "[no data]"
    nums = vals.str.extract(r"=\s*([0-9.]+)")[0].astype(float)
    top = nums.mode().iloc[0]
    lo, hi = nums.min(), nums.max()
    if lo == hi:
        return f"{top:g}"
    return f"{top:g} (range {lo:g}\u2013{hi:g})"


# ---------------------------------------------------------------- tables

def table4(main: Path):
    pm = load(main, "results_policy_per_machine.csv")
    if pm is None:
        table("Table 4 - Policy calibration",
              ["Policy", "Parameter calibrated", "Value selected (validation)",
               "Violation rate achieved (test) %", "Target %"],
              [["Predictive", "Margin multiplier k", "[no data]", "[no data]", "5.0"],
               ["Reactive", "Scale-up threshold", "[no data]", "[no data]", "5.0"]],
              note="results_policy_per_machine.csv not found - run with --policy")
        return
    rows = []
    for pol, label, param in [("predictive", "Predictive", "Margin multiplier k"),
                              ("reactive", "Reactive", "Scale-up threshold")]:
        sub = pm[pm["policy"] == pol]
        if sub.empty:
            rows.append([label, param, "[no data]", "[no data]", fmt(TARGET_DEFAULT, 1)])
            continue
        rows.append([label, param,
                     mode_and_range(sub["calibrated_param"]),
                     fmt(sub["sla_violation_pct"].mean()),
                     fmt(TARGET_DEFAULT, 1)])
    table("Table 4 - Policy calibration", 
          ["Policy", "Parameter calibrated", "Value selected (validation)",
           "Violation rate achieved (test) %", "Target %"], rows,
          note=f"mean over {pm['machine'].nunique()} machines and "
               f"{pm['provisioning_delay'].nunique()} provisioning delays")

    # per-delay detail, because one row hides the interesting part
    det = (pm.groupby(["provisioning_delay", "policy"])
             .agg(param=("calibrated_param", lambda s: mode_and_range(s)),
                  viol=("sla_violation_pct", "mean"))
             .reset_index())
    table("Table 4a - calibration by provisioning delay (supporting detail)",
          ["Delay", "Policy", "Value selected", "Violation achieved (test) %", "Target %"],
          [[int(r.provisioning_delay), r.policy.capitalize(), r.param,
            fmt(r.viol), "5.0"] for r in det.itertuples()])


def table5_and_6(main: Path):
    cmp_ = load(main, "results_policy_comparison.csv")
    if cmp_ is None:
        print("\n### Tables 5 and 6 - no results_policy_comparison.csv found")
        return
    order_pol = {"predictive": 0, "reactive": 1}
    cmp_ = (cmp_.assign(_p=cmp_["policy"].map(order_pol).fillna(9))
                .sort_values(["provisioning_delay", "_p"]))
    rows = [[int(r.provisioning_delay), r.policy.capitalize(),
             fmt(r.sla_violation_pct), fmt(r.mean_utilisation_pct),
             fmt(r.mean_idle_capacity), fmt(r.energy_proxy_wh, 1), fmt(r.cost_J)]
            for r in cmp_.itertuples()]
    table("Table 5 - Predictive against reactive allocation",
          ["Delay", "Policy", "SLA violation %", "Utilisation %",
           "Idle capacity", "Energy proxy (Wh)", "Cost J"], rows)

    label = {0: "0 intervals", 1: "1 interval (5 min)",
             3: "3 intervals (15 min)", 6: "6 intervals (30 min)"}
    rows6 = []
    for d, g in cmp_.groupby("provisioning_delay"):
        try:
            p = g[g["policy"] == "predictive"].iloc[0]
            r = g[g["policy"] == "reactive"].iloc[0]
        except IndexError:
            continue
        rows6.append([
            label.get(int(d), f"{int(d)} intervals"),
            f"+{p.mean_utilisation_pct - r.mean_utilisation_pct:.2f}",
            f"{p.mean_idle_capacity / r.mean_idle_capacity * 100:.1f}% of reactive"
            if r.mean_idle_capacity else "[no data]",
            f"{(p.energy_proxy_wh - r.energy_proxy_wh) / r.energy_proxy_wh * 100:+.2f}"
            if r.energy_proxy_wh else "[no data]",
            f"{(r.cost_J - p.cost_J) / r.cost_J * 100:.1f}% lower"
            if r.cost_J else "[no data]",
        ])
    table("Table 6 - Effect of provisioning delay on the predictive advantage",
          ["Provisioning delay", "Utilisation gain (pp)", "Idle capacity reduction",
           "Energy proxy change %", "Cost J reduction"], rows6)


def dedupe_by_window(dirs: list[Path]) -> dict[int, Path]:
    """One folder per residual window.

    A sweep folder usually sits inside the default output folder, so the same
    window can be found twice - once as outputs/ and once as outputs/out_w48.
    Prefer the folder whose own name encodes the window, because that one was
    certainly produced by the sweep.
    """
    chosen: dict[int, Path] = {}
    for d in dirs:
        w = read_window(d)
        if w is None:
            continue
        named = re.search(r"w(\d+)$", d.name) is not None
        if w not in chosen:
            chosen[w] = d
        elif named and re.search(r"w(\d+)$", chosen[w].name) is None:
            chosen[w] = d
    return chosen


def window_rows(dirs: list[Path]) -> list[dict]:
    """Per residual window: margin, violation, utilisation, idle and cost.

    Idle and the violation RATE are kept separately from cost_J so the cost
    weights can be swept afterwards instead of being baked in.
    """
    out = []
    for w, d in sorted(dedupe_by_window(dirs).items()):
        pm = load(d, "results_policy_per_machine.csv")
        if pm is not None and "policy" in pm.columns:
            sub = pm[pm["policy"] == "predictive"]
            if sub.empty:
                continue
            out.append({
                "window": w, "dir": d,
                "margin": sub["mean_margin"].mean(),
                "viol": sub["sla_violation_pct"].mean(),
                "util": sub["mean_utilisation_pct"].mean(),
                "idle": sub["mean_idle_capacity"].mean(),
                "cost": sub["cost_J"].mean(),
                "src": "calibrated predictive policy",
            })
            continue
        s_ = load(d, "results_summary.csv")
        if s_ is None or "mean_margin_mean" not in s_.columns:
            continue
        a, b = read_cost_weights(d)
        best = s_.sort_values("MAE_mean").iloc[0]
        out.append({
            "window": w, "dir": d,
            "margin": best["mean_margin_mean"],
            "viol": best["sla_violation_pct_mean"],
            "util": best["mean_utilisation_pct_mean"],
            "idle": best["mean_idle_capacity_mean"],
            "cost": a * best["mean_idle_capacity_mean"]
                    + b * best["sla_violation_pct_mean"] / 100,
            "src": "fixed-margin harness (uncalibrated)",
        })
    return out


WIN_LABEL = {12: "12 intervals (1 h)", 24: "24 intervals (2 h)",
             48: "48 intervals (4 h)", 96: "96 intervals (8 h)",
             192: "192 intervals (16 h)"}


def table7(dirs: list[Path]):
    rows = window_rows(dirs)
    if not rows:
        print("\n### Table 7 - no residual-window sweep found")
        return
    srcs = {r["src"] for r in rows}
    table("Table 7 - Residual-window sensitivity",
          ["Residual window", "Mean margin", "SLA violation %",
           "Utilisation %", "Cost J"],
          [[WIN_LABEL.get(r["window"], f"{r['window']} intervals"),
            fmt(r["margin"]), fmt(r["viol"]), fmt(r["util"]), fmt(r["cost"])]
           for r in rows],
          note="predictive policy, averaged over machines and delays; source: "
               + " and ".join(sorted(srcs)))

    best_v = min(rows, key=lambda r: r["viol"])
    best_c = min(rows, key=lambda r: r["cost"])
    print()
    print(f"- lowest violation rate: window {best_v['window']} "
          f"({best_v['viol']:.2f}%)")
    print(f"- lowest cost J: window {best_c['window']} "
          f"({best_c['cost']:.2f})")
    if best_v["window"] != best_c["window"]:
        print("- the two criteria disagree, so the choice of window depends on "
              "the cost weights - see Table 7a")


def table7a(dirs: list[Path]):
    """Which window wins as the price of an SLA breach rises.

    J(beta) = alpha * idle + beta * P(violation). Sweeping beta instead of
    quoting one number makes it clear that the window is a commercial choice
    rather than a result.
    """
    rows = window_rows(dirs)
    if len(rows) < 2:
        return
    alpha, _ = read_cost_weights(rows[0]["dir"])
    betas = [0, 1, 2, 5, 10, 20, 50, 100]
    out = []
    for b in betas:
        scored = [(alpha * r["idle"] + b * r["viol"] / 100.0, r) for r in rows]
        j, best = min(scored, key=lambda t: t[0])
        out.append([b, WIN_LABEL.get(best["window"], str(best["window"])),
                    fmt(j), fmt(best["viol"]), fmt(best["idle"])])
    table("Table 7a - which residual window wins as an SLA breach gets dearer",
          ["beta (price of one breach, in units of idle capacity)",
           "Window with the lowest J", "J at that window",
           "Its violation %", "Its idle capacity"], out,
          note="alpha = %g; the project's configured beta is %g"
               % (alpha, read_cost_weights(rows[0]["dir"])[1]))

    # exact flip point between the violation-minimising and the cheapest window
    bv = min(rows, key=lambda r: r["viol"])
    bc = min(rows, key=lambda r: alpha * r["idle"]
             + read_cost_weights(r["dir"])[1] * r["viol"] / 100.0)
    if bv["window"] != bc["window"]:
        d_idle = alpha * (bv["idle"] - bc["idle"])
        d_rate = (bv["viol"] - bc["viol"]) / 100.0
        if d_rate < 0:
            flip = d_idle / (-d_rate)
            print()
            print(f"- window {bv['window']} overtakes window {bc['window']} on "
                  f"cost once beta exceeds {flip:.1f}, i.e. once one SLA breach "
                  f"is priced above about {flip:.0f} units of idle capacity")


def table8_and_9(main: Path):
    s = load(main, "results_summary.csv")
    if s is None:
        print("\n### Tables 8 and 9 - no results_summary.csv found")
        return
    s = s.sort_values("MAE_mean")
    cfgf = main / "config_used.json"
    if cfgf.exists():
        try:
            asked = json.loads(cfgf.read_text()).get("models", [])
            missing = [m for m in asked if m not in set(s["model"])]
            if missing:
                print()
                print("> MISSING FORECASTERS: "
                      + ", ".join(PRETTY_MODEL.get(m, m) for m in missing)
                      + " were requested in the config but produced no results "
                        "in this run - the optional dependency is probably not "
                        "installed (arima needs statsmodels, lstm and gru need "
                        "torch). Either install it and rerun, or delete those "
                        "rows from the report rather than carrying numbers "
                        "over from a different run.")
        except Exception:
            pass
    order = ["naive", "seasonal_naive", "arima", "ridge", "hgb", "lstm", "gru"]
    s["_o"] = s["model"].apply(lambda m: order.index(m) if m in order else 99)
    s = s.sort_values("_o")

    rows8 = [[PRETTY_MODEL.get(r.model, r.model), fmt(r.MAE_mean, 4),
              fmt(getattr(r, "skill_vs_naive_pct_mean", float("nan"))),
              fmt(r.sla_violation_pct_mean), fmt(r.mean_utilisation_pct_mean)]
             for r in s.itertuples()]
    table("Table 8 - One fixed policy, different forecasters driving it",
          ["Forecaster driving the fixed policy", "Forecast MAE",
           "Skill vs naive %", "SLA violation %", "Utilisation %"], rows8,
          note="allocation policy and margin rule held constant; only the "
               "forecaster is swapped")

    budget_ms = 300_000.0
    rows9 = []
    for r in s.itertuples():
        inf = getattr(r, "predict_ms_per_row_mean", float("nan"))
        verdict = "[no data]"
        if not pd.isna(inf):
            at10k = inf * 10_000
            share = at10k / budget_ms * 100
            word = "Yes" if share < 100 else "No"
            verdict = f"{word} - {share:.1f}% of the budget at 10,000 machines"
        rows9.append([PRETTY_MODEL.get(r.model, r.model),
                      fmt(getattr(r, "fit_seconds_mean", float("nan")), 3),
                      fmt(inf, 4), verdict])
    table("Table 9 - Computational cost",
          ["Component", "Training time (s)", "Inference (ms per row)",
           "Within a 5-minute interval?"], rows9,
          note="a 5-minute interval is 300,000 ms; the verdict column scales "
               "one-row inference to a 10,000-machine estate")


# ---------------------------------------------------------------- main

class Tee:
    """Print to the screen and into report_tables.md at the same time."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        import sys as _s
        _s.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        import sys as _s
        _s.__stdout__.flush()
        self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder to search (default: here)")
    ap.add_argument("--main", default=None,
                    help="output dir of the default run (auto-detected otherwise)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    import sys
    out_md = root / "report_tables.md"
    sys.stdout = Tee(out_md)
    dirs = find_result_dirs(root)
    if not dirs:
        print(f"No results_*.csv found anywhere under {root}.")
        print("Run the experiment first:")
        print("  python run.py --config config/default.yaml")
        print("  python run.py --config config/default.yaml --policy")
        return

    print(f"# Phase-III report tables, filled from {root}")
    print()
    print("Result folders found:")
    for d in dirs:
        w = read_window(d)
        have = sorted(p.name.replace("results_", "").replace(".csv", "")
                      for p in d.glob("results_*.csv"))
        print(f"  - {d.relative_to(root) if d != root else '.'}  "
              f"(residual window: {w if w is not None else 'unknown'})  "
              f"[{', '.join(have)}]")

    if args.main:
        main_dir = Path(args.main).resolve()
    else:
        cands = [d for d in dirs if read_window(d) == 48] or \
                [d for d in dirs if d.name == "outputs"] or dirs
        # prefer whichever candidate has the most result files
        main_dir = max(cands, key=lambda d: len(list(d.glob("results_*.csv"))))
    print(f"\nUsing **{main_dir.relative_to(root) if main_dir != root else '.'}** "
          f"as the default run for Tables 4, 5, 6, 8, 9.")

    table4(main_dir)
    table5_and_6(main_dir)
    table7(dirs)
    table7a(dirs)
    table8_and_9(main_dir)

    print()
    print("---")
    print("Any cell reading [no data] has nothing behind it in the CSVs. "
          "Leave it empty in the report or say in the text why it is absent - "
          "do not fill it by hand.")
    sys.stdout.flush()
    sys.__stdout__.write(f"\nAlso saved to {out_md}\n")


if __name__ == "__main__":
    main()
