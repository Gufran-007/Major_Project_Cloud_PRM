"""
The frozen forecaster -> allocator contract (Phase-II report, Section 3.4).

WHY THIS IS A SEPARATE FILE WITH A DATACLASS
--------------------------------------------
The allocator is allowed to see EXACTLY these four fields and nothing else:

    demand      the predicted demand over the horizon
    dispersion  a scalar summarising how reliable this forecaster has been
                lately (from residuals.py)
    issued_at   when the forecast was made
    model_id    which model produced it - for logging ONLY

It deliberately has NO access to model internals, feature importances, or
training data. If the policy could inspect the model, model-specific tuning
would leak into the allocation comparison and quietly invalidate the whole
result. Freezing this boundary early is what makes the Phase-III measurement
trustworthy, so treat changes to this class as a decision, not a refactor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from .residuals import ResidualTracker


@dataclass(frozen=True)
class Forecast:
    demand: List[float]
    dispersion: float
    issued_at: pd.Timestamp
    model_id: str

    def committed_capacity(self, k: float = 1.5) -> List[float]:
        """What the allocator would provision. The ONLY method the policy calls."""
        return [d + k * self.dispersion for d in self.demand]


class ForecastService:
    """Wraps any forecaster and emits Forecast objects.

    This is the seam the project is built around: everything above it can be
    swapped freely, everything below it never changes.
    """

    def __init__(self, model, model_id: str, residual_window: int = 48, k: float = 1.5):
        self.model = model
        self.model_id = model_id
        self.k = k
        self._tracker = ResidualTracker(window=residual_window, k=k)

    def forecast_row(self, x_row: pd.DataFrame, issued_at: pd.Timestamp) -> Forecast:
        demand = self.model.predict(x_row)
        return Forecast(demand=[float(v) for v in demand],
                        dispersion=self._tracker.dispersion(),
                        issued_at=issued_at,
                        model_id=self.model_id)

    def observe(self, y_true: float, y_pred: float) -> None:
        """Feed the realised value back so the dispersion estimate stays current."""
        self._tracker.update(y_true, y_pred)
