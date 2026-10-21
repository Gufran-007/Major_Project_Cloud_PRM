"""
Every forecaster, behind one interface.

THE INTERFACE IS THE POINT
--------------------------
Phase-I's central claim is that you can hold the allocation policy fixed and
swap ONLY the forecaster, then attribute any change in outcome to prediction
quality. That is only true if every model is genuinely interchangeable. So all
of them implement exactly:

    fit(X_train, y_train, X_val, y_val) -> self
    predict(X) -> np.ndarray

Nothing downstream is allowed to know which model it is holding. If you ever
find yourself writing `if isinstance(model, LSTM)` outside this file, stop -
that is the comparison starting to leak.

Heavy dependencies (statsmodels, lightgbm, torch) are OPTIONAL. If they are not
installed the model is skipped with a message rather than crashing the run, so
the code still works on a fresh laptop.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge


class BaseForecaster:
    """Common interface. Subclasses override _fit/_predict only."""
    name = "base"
    needs_scaling = False           # tree models do not care; linear models do

    def __init__(self, **kw):
        self.params = kw
        self.fit_seconds: float = 0.0
        self.predict_ms_per_row: float = 0.0

    # -- public API (do not override) ------------------------------------
    def fit(self, X_tr, y_tr, X_va=None, y_va=None):
        t0 = time.perf_counter()
        self._fit(X_tr, y_tr, X_va, y_va)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def predict(self, X) -> np.ndarray:
        t0 = time.perf_counter()
        out = self._predict(X)
        elapsed = time.perf_counter() - t0
        self.predict_ms_per_row = (elapsed / max(1, len(X))) * 1000
        return np.asarray(out, dtype=float).ravel()

    # -- to implement ------------------------------------------------------
    def _fit(self, X_tr, y_tr, X_va, y_va): raise NotImplementedError
    def _predict(self, X): raise NotImplementedError


# ======================================================================
# BASELINES - the models everything else must beat to justify itself
# ======================================================================

class NaivePersistence(BaseForecaster):
    """Predict that the next value equals the current one.

    Sounds trivial; on 5-minute cloud traces it is a genuinely strong baseline
    and many published deep-learning results barely beat it. If your LSTM
    cannot beat this, that is the finding - report it.
    """
    name = "naive"

    def _fit(self, X_tr, y_tr, X_va, y_va):
        self.col_ = self._lag1_column(X_tr)
        return self

    @staticmethod
    def _lag1_column(X) -> str:
        """The most recent observation available: lag 0 if present, else lag 1."""
        for suffix in ("_lag0", "_lag1"):
            for c in X.columns:
                if c.endswith(suffix) and c.startswith("cpu"):
                    return c
        cands = [c for c in X.columns if "_lag" in c]
        return min(cands, key=lambda c: int(c.rsplit("lag", 1)[1]))

    def _predict(self, X):
        return X[self.col_].to_numpy()


class SeasonalNaive(BaseForecaster):
    """Predict the value from one season ago (default: 24 intervals = 2 hours).

    Uses whichever long lag is available. Captures daily rhythm cheaply.
    """
    name = "seasonal_naive"

    def _fit(self, X_tr, y_tr, X_va, y_va):
        cands = [c for c in X_tr.columns if c.startswith("cpu") and "_lag" in c]
        self.col_ = max(cands, key=lambda c: int(c.rsplit("lag", 1)[1]))
        return self

    def _predict(self, X):
        return X[self.col_].to_numpy()


class RidgeForecaster(BaseForecaster):
    """Regularised linear regression on the full feature table.

    The honest 'classical machine learning' baseline. Fast, stable, and
    surprisingly hard to beat by a wide margin on this kind of data.
    """
    name = "ridge"
    needs_scaling = True

    def _fit(self, X_tr, y_tr, X_va, y_va):
        best, best_mae = None, np.inf
        # Tiny validation sweep. Baselines get a tuning budget too - a
        # comparison against an untuned baseline is not a comparison.
        for alpha in (0.1, 1.0, 10.0, 100.0):
            m = Ridge(alpha=alpha).fit(X_tr, y_tr)
            if X_va is None or len(X_va) == 0:
                best = m; break
            mae = float(np.mean(np.abs(m.predict(X_va) - y_va)))
            if mae < best_mae:
                best, best_mae = m, mae
        self.model_ = best
        return self

    def _predict(self, X):
        return self.model_.predict(X)


# ======================================================================
# TREE ENSEMBLES
# ======================================================================

class HistGBForecaster(BaseForecaster):
    """Histogram gradient boosting from scikit-learn.

    Chosen over XGBoost/LightGBM as the DEFAULT because it ships with sklearn,
    so this file has no extra install step. If you have LightGBM, use that
    instead - it is usually a little faster - but expect similar accuracy.
    """
    name = "hgb"

    def _fit(self, X_tr, y_tr, X_va, y_va):
        best, best_mae = None, np.inf
        for lr, leaves in ((0.05, 31), (0.1, 31), (0.1, 63)):
            m = HistGradientBoostingRegressor(
                learning_rate=lr, max_leaf_nodes=leaves,
                max_iter=300, early_stopping=True, random_state=0,
            ).fit(X_tr, y_tr)
            if X_va is None or len(X_va) == 0:
                best = m; break
            mae = float(np.mean(np.abs(m.predict(X_va) - y_va)))
            if mae < best_mae:
                best, best_mae = m, mae
        self.model_ = best
        return self

    def _predict(self, X):
        return self.model_.predict(X)


class RandomForestForecaster(BaseForecaster):
    name = "rf"

    def _fit(self, X_tr, y_tr, X_va, y_va):
        self.model_ = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=0
        ).fit(X_tr, y_tr)
        return self

    def _predict(self, X):
        return self.model_.predict(X)


# ======================================================================
# OPTIONAL MODELS - only available if the dependency is installed
# ======================================================================

class ArimaForecaster(BaseForecaster):
    """ARIMA on the target series alone (needs statsmodels).

    Note it ignores the feature table by construction - it is a univariate
    method. That is exactly why it belongs in the comparison: it shows how much
    the extra features are worth.
    """
    name = "arima"

    def _fit(self, X_tr, y_tr, X_va, y_va):
        from statsmodels.tsa.arima.model import ARIMA
        self.ARIMA = ARIMA
        self.history_ = np.asarray(y_tr, dtype=float)
        self.order_ = (2, 0, 1)
        self.res_ = ARIMA(self.history_, order=self.order_).fit()
        return self

    def _predict(self, X):
        """ROLLING one-step-ahead forecasts.

        This matters for fairness. A single forecast(steps=n) call would be an
        n-step-ahead forecast, which for n=800 is hopeless and would make ARIMA
        look absurdly bad next to models that only ever predict one step. Here
        we forecast one step, then feed in the observation that actually
        occurred, exactly as the other models implicitly do through their lag
        features. refit=False keeps the fitted coefficients and only extends the
        state, so this stays fast.
        """
        obs = self._observed_history(X)
        res = self.res_
        preds = []
        for i in range(len(X)):
            preds.append(float(np.asarray(res.forecast(steps=1))[0]))
            if i < len(obs):
                res = res.append([obs[i]], refit=False)
        return np.asarray(preds, dtype=float)

    @staticmethod
    def _observed_history(X) -> np.ndarray:
        """Recover the realised series from the lag-0 (or lag-1) feature column."""
        for suffix in ("_lag0", "_lag1"):
            for c in X.columns:
                if c.endswith(suffix) and c.startswith("cpu"):
                    return X[c].to_numpy(dtype=float)
        return X.iloc[:, 0].to_numpy(dtype=float)


class LSTMForecaster(BaseForecaster):
    """Small LSTM over the feature vector (needs PyTorch).

    Deliberately modest: 1 layer, 64 hidden units. The literature's gains come
    mostly from architecture search you do not have the compute budget for, and
    a small honest model beats a large badly-trained one.
    """
    name = "lstm"
    needs_scaling = True

    def _fit(self, X_tr, y_tr, X_va, y_va):
        import torch                                        # noqa: F401
        import torch.nn as nn

        torch.manual_seed(0)
        self.torch = torch
        Xt = torch.tensor(X_tr.to_numpy(), dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(np.asarray(y_tr), dtype=torch.float32).view(-1, 1)

        class Net(nn.Module):
            def __init__(self, n_in):
                super().__init__()
                self.rnn = nn.LSTM(n_in, 64, batch_first=True)
                self.head = nn.Linear(64, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        self.model_ = Net(Xt.shape[-1])
        opt = torch.optim.Adam(self.model_.parameters(), lr=1e-3)
        lossf = nn.L1Loss()                     # L1 matches the MAE we report
        ds = torch.utils.data.TensorDataset(Xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
        for _ in range(15):
            for xb, yb in dl:
                opt.zero_grad(); lossf(self.model_(xb), yb).backward(); opt.step()
        self.model_.eval()
        return self

    def _predict(self, X):
        t = self.torch
        with t.no_grad():
            xb = t.tensor(X.to_numpy(), dtype=t.float32).unsqueeze(1)
            return self.model_(xb).numpy().ravel()


# ======================================================================
# REGISTRY - the only place a model name maps to a class
# ======================================================================

REGISTRY = {
    "naive": NaivePersistence,
    "seasonal_naive": SeasonalNaive,
    "ridge": RidgeForecaster,
    "hgb": HistGBForecaster,
    "rf": RandomForestForecaster,
    "arima": ArimaForecaster,
    "lstm": LSTMForecaster,
}

OPTIONAL_DEPS = {"arima": "statsmodels", "lstm": "torch"}


def build(names: List[str]) -> Dict[str, BaseForecaster]:
    """Instantiate the requested models, skipping any whose dependency is absent."""
    out: Dict[str, BaseForecaster] = {}
    for n in names:
        if n not in REGISTRY:
            print(f"[models] unknown model '{n}' - skipping. "
                  f"Known: {sorted(REGISTRY)}")
            continue
        dep = OPTIONAL_DEPS.get(n)
        if dep:
            try:
                __import__(dep)
            except ImportError:
                print(f"[models] '{n}' needs {dep}, which is not installed - skipping. "
                      f"Install it with:  pip install {dep}")
                continue
        out[n] = REGISTRY[n]()
    return out
