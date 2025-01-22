import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model

def adf_stationarity_test(series, title="ADF Test"):
    """
    Performs the Augmented Dickey-Fuller test for stationarity on a Pandas Series
    and prints the test statistic, p-value, and conclusion.
    """
    print(f"\n=== {title} ===")
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    for key, value in result[4].items():
        print('Critical Values:')
        print(f"   {key}, {value}")
    if result[1] < 0.05:
        print("=> Stationary (reject H0)\n")
    else:
        print("=> Not Stationary (fail to reject H0)\n")

def plot_acf_pacf(series, lags=30):
    """
    Plots the ACF and PACF for a given series to help identify AR and MA orders.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0], lags=lags, title="ACF")
    plot_pacf(series.dropna(), ax=axes[1], lags=lags, title="PACF")
    plt.show()

def make_stationary(series):
    """
    Simple differencing if the series is non-stationary. Returns the differenced series.
    (In practice, you'd check if differencing is actually required, perhaps multiple times.)
    """
    return series.diff(1).dropna()

def fit_sarima(train_series, order=(1,1,1), seasonal_order=(0,0,0,0)):
    """
    Fits a SARIMAX model (SARIMA if seasonal_order is specified) on the training series.
    Returns the fitted model.
    """
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False)
    fitted_model = model.fit(disp=False)
    return fitted_model

def fit_garch(train_series, p=1, q=1):
    """
    Fits a GARCH(p,q) model using the 'arch' library on the given train_series.
    Typically the series should be returns (or residuals), not raw price.
    Returns the fitted model.
    """
    # Drop NaNs
    series_clean = train_series.dropna()
    model = arch_model(series_clean, p=p, q=q, mean='Zero', vol='GARCH', dist='normal')
    res = model.fit(disp='off')
    return res

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
