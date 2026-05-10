"""Standalone ARIMA(1,1,1) sales predictor.

A small wrapper around ``statsmodels`` ARIMA that fits a model on the
supplied history and exposes ``predict`` / ``update`` for online use.
Not currently wired into the main simulator loop (``BaselinePolicy``
uses a simple rolling-mean trend instead). Kept available for
downstream / notebook analysis or as a forecasting alternative to plug
into a custom ``Policy``.
"""

# NumPy is used for the in-place ``append`` in ``update``.
import numpy as np
# ARIMA implementation from statsmodels — heavy dependency, hence why
# this module is *not* imported by ``src/__init__.py``.
from statsmodels.tsa.arima.model import ARIMA


class SalesPredictor:
    """One-process, one-product ARIMA(1,1,1) forecaster.

    Holds a single fitted model. Call ``update(new_data)`` to append
    observations and refit; ``predict(steps)`` returns a point forecast
    array.
    """

    def __init__(self, history):
        """Initialize predictor with a historical sales series.

        ``history`` is anything 1-D-array-like that ARIMA accepts —
        typically a NumPy array, list, or pandas Series of integer sales
        counts.
        """
        # Cached historical observations; appended-to in ``update``.
        self.history = history
        # Initial fit; refit on every ``update``.
        self.model = self.fit()

    def fit(self):
        """Fit and return an ARIMA(1,1,1) model on the current history."""
        # ``order=(p, d, q)`` — one AR lag, one differencing, one MA lag.
        # Small order keeps fits fast and avoids over-parameterising on
        # short histories.
        return ARIMA(self.history, order=(1, 1, 1)).fit()

    def predict(self, steps):
        """Forecast future sales values for the requested number of steps."""
        return self.model.forecast(steps)

    def update(self, new_data):
        """Append new observations and refit the forecasting model."""
        # ``np.append`` always copies — fine for the small histories
        # this class is used on.
        self.history = np.append(self.history, new_data)
        self.model = self.fit()
