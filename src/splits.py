"""
Chronological splitting and scaling.

WHY NOT train_test_split(shuffle=True)?
---------------------------------------
Because it would let the model train on Wednesday and be tested on Tuesday.
Time series must be split by time, full stop. Everything before a cut-off is
training; everything after is unseen future.

The scaler is fitted on the TRAINING SLICE ONLY and then applied unchanged to
validation and test. Fitting it on everything would leak the future's mean and
variance backwards into training.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import Config


@dataclass
class Split:
    X_train: pd.DataFrame; y_train: pd.Series
    X_val:   pd.DataFrame; y_val:   pd.Series
    X_test:  pd.DataFrame; y_test:  pd.Series

    def sizes(self) -> str:
        return (f"train={len(self.y_train)}  val={len(self.y_val)}  "
                f"test={len(self.y_test)}")


def chronological_split(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Split:
    n = len(X)
    i_tr = int(n * cfg.train_frac)
    i_va = int(n * (cfg.train_frac + cfg.val_frac))
    return Split(
        X.iloc[:i_tr], y.iloc[:i_tr],
        X.iloc[i_tr:i_va], y.iloc[i_tr:i_va],
        X.iloc[i_va:], y.iloc[i_va:],
    )


def scale(split: Split) -> Split:
    """Standardise features using training-set statistics only."""
    sc = StandardScaler().fit(split.X_train)
    f = lambda d: pd.DataFrame(sc.transform(d), index=d.index, columns=d.columns)
    return Split(f(split.X_train), split.y_train,
                 f(split.X_val), split.y_val,
                 f(split.X_test), split.y_test)
