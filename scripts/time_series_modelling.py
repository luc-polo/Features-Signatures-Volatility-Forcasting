import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from pmdarima import auto_arima
import itertools
import warnings

def plot_acf_pacf(series, lags=30):
    """
    Plots the ACF and PACF for a given series to help identify AR and MA orders.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0], lags=lags, title="ACF")
    plot_pacf(series.dropna(), ax=axes[1], lags=lags, title="PACF")
    plt.show()

def differentiate_ts(series):
    """
    Simple differencing if the series is non-stationary. Returns the differenced series.
    (In practice, you'd check if differencing is actually required, perhaps multiple times.)
    """
    return series.diff(1).dropna()

def fit_sarima(train_series):
    """
    Automatically fits a SARIMA model to the training series using auto_arima.
    Returns the fitted model.
    """
    auto_model = auto_arima(train_series, seasonal=True, trace=False, stepwise=True, suppress_warnings=True)
    model = SARIMAX(train_series, order=auto_model.order, seasonal_order=auto_model.seasonal_order, enforce_stationarity=False)
    return model.fit(disp=False)

def fit_garch_auto(train_series, p_max=5, q_max=5, criterion='aic'):
    """
    Automatically fits a GARCH(p, q) model by selecting the best p and q based on the specified information criterion.
    
    Parameters:
    -----------
    train_series : pandas.Series
        The training time series data (typically returns or residuals).
    p_max : int, optional (default=5)
        The maximum lag order for the GARCH model's volatility equation.
    q_max : int, optional (default=5)
        The maximum lag order for the GARCH model's volatility equation.
    criterion : str, optional (default='aic')
        The information criterion to use for model selection. Options are 'aic' or 'bic'.
    
    Returns:
    --------
    best_model : arch.univariate.base.ARCHModelResult
        The fitted GARCH model with the lowest specified information criterion.
    best_p : int
        The selected GARCH(p, q) order for p.
    best_q : int
        The selected GARCH(p, q) order for q.
    
    Raises:
    -------
    ValueError:
        If an invalid criterion is specified or no valid model is found within the specified p and q ranges.
    """
    if criterion.lower() not in ['aic', 'bic']:
        raise ValueError("Criterion must be either 'aic' or 'bic'.")

    series_clean = train_series.dropna()
    best_score = np.inf
    best_model = None
    best_p, best_q = 0, 0

    # Suppress warnings temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, q in itertools.product(range(1, p_max + 1), range(1, q_max + 1)):
            try:
                model = arch_model(series_clean, mean='Zero', vol='GARCH', p=p, q=q, dist='normal')
                res = model.fit(disp='off')
                score = res.aic if criterion.lower() == 'aic' else res.bic
                if score < best_score:
                    best_score = score
                    best_model = res
                    best_p, best_q = p, q
            except Exception:
                continue  # Skip invalid models

    if best_model is None:
        raise ValueError("No valid GARCH model found within the specified p and q ranges.")

    print(f"Best GARCH model: GARCH({best_p}, {best_q}) with {criterion.upper()} = {best_score:.2f}")
    return best_model, best_p, best_q

def multi_step_forecast_sarima(fitted_model, steps):
    """
    Forecast `steps` periods ahead using the SARIMA fitted_model.
    Returns a Series of predictions aligned by date if the model has a proper index.
    """
    forecast = fitted_model.forecast(steps=steps)
    # Depending on statsmodels version, we might get .predicted_mean
    return forecast.predicted_mean

def compute_volatility(time_series, window=21):
    """
    Computes an annualized rolling standard deviation (volatility) 
    using a window of size 'window'.
    Returns a Pandas Series of volatilities.
    """
    daily_std = time_series.rolling(window=window).std()
    # Simple annualization factor example (assuming ~252 trading days)
    ann_vol = daily_std * np.sqrt(252)
    return ann_vol
