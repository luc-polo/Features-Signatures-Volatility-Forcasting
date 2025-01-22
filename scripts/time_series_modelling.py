import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from arch import arch_model
import itertools
import warnings
from itertools import product
from typing import Tuple


def plot_acf_pacf(series, lags=30):
    """
    Plots the ACF and PACF for a given series to help identify AR and MA orders.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series.dropna(), ax=axes[0], lags=lags, title="ACF")
    plot_pacf(series.dropna(), ax=axes[1], lags=lags, title="PACF")
    plt.show()

def differentiate_ts(series, diff=1):
    """
    Differencing the series 'diff' times. Returns the differenced series.
    """
    for _ in range(diff):
        series = series.diff().dropna()
    return series



def fit_arima(train_series, p, d, q):
    """
    Ajuste un modèle ARIMA avec les paramètres spécifiés et renvoie le modèle entraîné.

    Parameters:
        train_series (pd.Series): La série temporelle d'entraînement.
        p (int): Ordre AR (Auto-Régressif).
        d (int): Ordre de différenciation.
        q (int): Ordre MA (Moyenne Mobile).

    Returns:
        model_fit (ARIMAResults): Le modèle ARIMA ajusté.
    """
    try:
        # Ajuster le modèle ARIMA
        model = ARIMA(train_series, order=(p, d, q))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignorer les avertissements durant l'ajustement
            model_fit = model.fit()
        print(f"Modèle ARIMA({p}, {d}, {q}) ajusté avec succès.")
        return model_fit
    except Exception as e:
        print(f"Erreur lors de l'ajustement du modèle ARIMA({p}, {d}, {q}): {e}")
        return None



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
