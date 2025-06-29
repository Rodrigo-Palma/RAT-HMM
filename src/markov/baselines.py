# src/markov/baselines.py

import numpy as np
import pandas as pd
import statsmodels.api as sm

def moving_average_forecast(series, window_size):
    """
    Retorna a previsão para a série usando uma média móvel simples.
    """
    return series.rolling(window=window_size).mean().shift(1)

def arima_forecast(series, order=(1, 0, 0), steps=1):
    """
    Retorna a previsão usando um modelo ARIMA simples.
    """
    model = sm.tsa.ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def constant_mean_forecast(series):
    """
    Retorna a previsão como sendo a média da série.
    """
    mean_value = series.mean()
    forecast = np.full(shape=(len(series),), fill_value=mean_value)
    return forecast
