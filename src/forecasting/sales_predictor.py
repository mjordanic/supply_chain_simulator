import numpy as np
from statsmodels.tsa.arima.model import ARIMA


class SalesPredictor:
    def __init__(self, history):
        """Initialize predictor with historical sales series."""
        self.history = history
        self.model = self.fit()

    def fit(self):
        """Fit and return an ARIMA(1,1,1) model on current history."""
        return ARIMA(self.history, order=(1, 1, 1)).fit()

    def predict(self, steps):
        """Forecast future sales values for the requested number of steps."""
        return self.model.forecast(steps)

    def update(self, new_data):
        """Append new observations and refit the forecasting model."""
        self.history = np.append(self.history, new_data)
        self.model = self.fit()
