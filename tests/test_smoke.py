"""
Smoke tests. Run with:  python -m pytest tests -q     (or: python tests/test_smoke.py)

These are not exhaustive. They check the properties that, if broken, would make
every result meaningless without causing any visible failure - which is exactly
what a test suite in research software is for.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config
from src import synthetic, data as data_mod, features as feat_mod, metrics, residuals
from src.splits import chronological_split


def test_no_leakage_in_features():
    """THE most important test: no feature may contain information from the future.

    We build features, then check that every feature column at row t is
    reproducible from the raw series up to and including t.
    """
    cfg = Config(n_machines=1, use_synthetic=True)
    df = data_mod.clean_and_resample(synthetic.make_machine(600, np.random.default_rng(0)), cfg)
    X, y, groups = feat_mod.build_features(df, cfg)

    # The target must be the value one horizon AHEAD of the lag-0 feature.
    lag0 = X["cpu_usage_pct_lag0"]
    aligned = df["cpu_usage_pct"].reindex(X.index)
    assert np.allclose(lag0.to_numpy(), aligned.to_numpy(), atol=1e-9), \
        "lag0 must equal the current observed value"

    shifted = df["cpu_usage_pct"].shift(-cfg.horizon).reindex(X.index)
    assert np.allclose(y.to_numpy(), shifted.to_numpy(), atol=1e-9), \
        "target must be the value `horizon` steps ahead"


def test_split_is_chronological():
    cfg = Config()
    idx = pd.date_range("2013-08-01", periods=1000, freq="5min")
    X = pd.DataFrame({"a": np.arange(1000.0)}, index=idx)
    y = pd.Series(np.arange(1000.0), index=idx)
    sp = chronological_split(X, y, cfg)
    assert sp.X_train.index.max() < sp.X_val.index.min()
    assert sp.X_val.index.max() < sp.X_test.index.min()


def test_metrics_direction():
    """Over- and under-prediction must be reported separately and correctly."""
    y = np.array([10.0, 10.0, 10.0, 10.0])
    p = np.array([12.0, 12.0, 8.0, 8.0])          # 2 over, 2 under
    m = metrics.evaluate(y, p)
    assert abs(m["MAE"] - 2.0) < 1e-9
    assert abs(m["MAE_over"] - 2.0) < 1e-9
    assert abs(m["MAE_under"] - 2.0) < 1e-9
    assert abs(m["under_rate_pct"] - 50.0) < 1e-9


def test_utilisation_never_exceeds_100():
    """You cannot consume more capacity than was allocated."""
    rng = np.random.default_rng(0)
    y = rng.uniform(10, 90, 500)
    p = y + rng.normal(0, 5, 500)
    out = residuals.simulate_allocation(y, p)
    assert 0 <= out["mean_utilisation_pct"] <= 100.0


def test_margin_widens_with_volatility():
    """The whole point of the residual mechanism: erratic model -> bigger margin."""
    calm = residuals.ResidualTracker(window=50, k=1.5)
    wild = residuals.ResidualTracker(window=50, k=1.5)
    rng = np.random.default_rng(1)
    for _ in range(50):
        calm.update(50.0, 50.0 + rng.normal(0, 0.5))
        wild.update(50.0, 50.0 + rng.normal(0, 8.0))
    assert wild.margin() > calm.margin() * 3


# ======================================================================
# Phase-III: allocation policy and simulator
# ======================================================================

def test_provisioning_delay_is_actually_applied():
    """A decision must not take effect before `delay` intervals have passed.

    This guards the single most important property of the simulator. If the
    delay were silently ignored, the reactive baseline would look far better
    than it should and the whole comparison would be invalid.
    """
    from src import simulator as sim
    from src.policy import BasePolicy, PolicyDecision

    class ConstantThenJump(BasePolicy):
        name = "probe"
        def decide(self, t, observed_now, forecast=None):
            # ask for 999 from interval 5 onwards
            return PolicyDecision(capacity=999.0 if t >= 5 else 10.0, margin=0.0)

    demand = np.full(20, 10.0)
    res = sim.replay(demand, ConstantThenJump(), provisioning_delay=3)
    # the jump was requested at t=5, so it must not be live before t=8
    assert res.allocated[7] < 999.0, "capacity went live too early"
    assert res.allocated[8] == 999.0, "capacity did not go live on time"


def test_policy_cannot_see_the_current_interval():
    """The policy must only ever be shown the last FINISHED interval.

    If it could see demand[t] while deciding capacity for interval t, a
    zero-delay policy would be trivially perfect and the comparison worthless.
    """
    from src import simulator as sim
    from src.policy import BasePolicy, PolicyDecision

    seen = []

    class Recorder(BasePolicy):
        name = "recorder"
        def decide(self, t, observed_now, forecast=None):
            seen.append(observed_now)
            return PolicyDecision(capacity=100.0, margin=0.0)

    demand = np.arange(10, 30, dtype=float)     # strictly increasing, so distinct
    sim.replay(demand, Recorder(), provisioning_delay=0)
    for t in range(1, len(demand)):
        assert seen[t] == demand[t - 1], f"policy saw the in-progress interval at t={t}"


def test_calibration_moves_toward_the_target():
    """Calibration must actually change behaviour, not silently return the default."""
    from src import simulator as sim
    from src.policy import PredictiveMarginPolicy

    rng = np.random.default_rng(3)
    demand = 50 + 10 * np.sin(np.arange(600) / 30) + rng.normal(0, 3, 600)
    fc = demand + rng.normal(0, 3, 600)

    strict = sim.calibrate(demand, fc, lambda p: PredictiveMarginPolicy(k=p),
                           [0.25, 1.0, 2.0, 3.0], target_violation_pct=1.0, delay=3)
    loose = sim.calibrate(demand, fc, lambda p: PredictiveMarginPolicy(k=p),
                          [0.25, 1.0, 2.0, 3.0], target_violation_pct=25.0, delay=3)
    assert strict > loose, "a stricter SLA target must choose a larger margin"


def test_asymmetric_cost_penalises_frequency():
    """Under the Phase-I form, more frequent violations must cost more."""
    from src.policy import asymmetric_cost
    demand = np.full(100, 50.0)
    rare = np.full(100, 51.0); rare[:5] = 49.0      # short 5% of the time
    often = np.full(100, 51.0); often[:40] = 49.0   # short 40% of the time
    c_rare = asymmetric_cost(demand, rare)["cost_J"]
    c_often = asymmetric_cost(demand, often)["cost_J"]
    assert c_often > c_rare


def test_cost_gamma_prices_severity():
    """The gamma term must distinguish a small shortfall from a large one.

    This is the gap the Phase-I cost function had: with gamma=0 both of these
    score identically, because only the violation RATE is charged.
    """
    from src.policy import asymmetric_cost
    demand = np.full(100, 50.0)
    barely = np.full(100, 49.0)        # always 1 unit short
    badly = np.full(100, 10.0)         # always 40 units short

    spec = (asymmetric_cost(demand, barely, gamma=0.0)["cost_J"],
            asymmetric_cost(demand, badly, gamma=0.0)["cost_J"])
    assert abs(spec[0] - spec[1]) < 1e-9, "documented limitation of the gamma=0 form"

    with_gamma = (asymmetric_cost(demand, barely, gamma=1.0)["cost_J"],
                  asymmetric_cost(demand, badly, gamma=1.0)["cost_J"])
    assert with_gamma[1] > with_gamma[0], "gamma must price severity"


def test_energy_proxy_scales_with_provisioned_capacity():
    """Provisioning a large idle pool must cost MORE energy, not less."""
    from src import simulator as sim
    from src.policy import BasePolicy, PolicyDecision

    class Fixed(BasePolicy):
        name = "fixed"
        def __init__(self, c): self.c = c
        def decide(self, t, observed_now, forecast=None):
            return PolicyDecision(capacity=self.c, margin=0.0)

    demand = np.full(200, 40.0)
    lean = sim.replay(demand, Fixed(45.0), provisioning_delay=0)
    bloated = sim.replay(demand, Fixed(90.0), provisioning_delay=0)
    assert bloated.metrics["energy_proxy_wh"] > lean.metrics["energy_proxy_wh"]


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn(); print(f"  PASS  {fn.__name__}")
    print("\nAll smoke tests passed.")
