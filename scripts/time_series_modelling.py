import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import itertools
import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error




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


def fit_garch(series, p, q, mean="AR", vol="GARCH", dist="t"):
    """
    Train a GARCH model with specified parameters.

    Parameters:
        series (pd.Series): Time series data (e.g., residuals from ARIMA).
        p (int): Order of the GARCH component.
        q (int): Order of the ARCH component.
        mean (str): Mean model. Default is "Zero".
        vol (str): Volatility model. Default is "GARCH".
        dist (str): Distribution of errors. Default is "normal".

    Returns:
        fitted_model: The fitted GARCH model.
    """
    garch = arch_model(series, mean=mean, lags=1, vol=vol, p=p, q=q, dist=dist)

    fitted_model = garch.fit(disp="off")
    return fitted_model


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
        for p, q in itertools.product(range(0, p_max + 1), range(0, q_max + 1)):
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

def forecast_arima(fitted_arima, steps=21):
    """
    Forecasts 'steps' future values using the ARIMA model.
    Returns a numpy array of predictions.
    """
    forecast_result = fitted_arima.forecast(steps=steps)
    
    # Si le retour est une série, pas besoin d'accéder à predicted_mean
    if isinstance(forecast_result, pd.Series):
        return forecast_result.values
    # Si le retour est un objet complexe, accédez à predicted_mean
    return forecast_result.predicted_mean.values

def forecast_garch(fitted_garch, steps=21):
    """
    Forecasts the conditional volatility (std) for 'steps' ahead from the GARCH model.
    Returns a numpy array of predicted daily standard deviations.
    """
    garch_forecast = fitted_garch.forecast(horizon=steps)
    # 'variance' is often returned. We take its square root for volatility:
    # Typically: garch_forecast.variance[-1] is the last forecast block
    predicted_var = garch_forecast.variance.iloc[-1].values
    predicted_vol = np.sqrt(predicted_var)
    return predicted_vol


def combine_arima_garch(arima_preds, garch_vol):
    """
    Example formula combining ARIMA mean predictions and GARCH vol forecasts
    to estimate daily absolute moves.
    
    One simple approach (among many possible) is to assume for each day t:
    
    predicted_absolute_move[t] = sqrt( (X_{t} - X_{t-1})^2 + garch_vol[t]^2 )
    
    where X_{t} comes from ARIMA, and garch_vol[t] is the predicted std dev from GARCH.
    
    Returns a numpy array of size len(arima_preds) - 1.
    """
    abs_moves = []
    for i in range(1, len(arima_preds)):
        mean_diff_sq = (arima_preds[i] - arima_preds[i-1]) ** 2
        abs_move = np.sqrt(mean_diff_sq + garch_vol[i]**2)
        abs_moves.append(abs_move)
    return np.array(abs_moves)

def estimate_rolling_volatility(abs_moves):
    """
    Estimate annualized volatility over a given rolling window of daily absolute moves.
    
    Parameters:
        abs_moves (list or np.ndarray): Predicted absolute daily moves or returns.
        
    Returns:
        float: The annualized volatility forecast for the given window size.
    """
    # Compute standard deviation over the window
    daily_std = np.std(abs_moves, ddof=1)
    
    # Annualize the volatility
    annualized_vol = daily_std * np.sqrt(252)
    return annualized_vol


def forecast_volatility(arima_order, garch_order, df, test_start, realized_vol, steps=21):
    """
    Forecasts volatility using pre-trained ARIMA and GARCH models.

    Parameters:
        arima_model (ARIMA): Pre-trained ARIMA model.
        garch_model (ARCHModelResult): Pre-trained GARCH model.
        df (pd.DataFrame): DataFrame containing the 'Close' price with datetime index.
        test_start (pd.Timestamp): Start date for the test set.
        realized_vol (pd.Series): Series of realized volatilities indexed by date.
        garch_order (tuple): (p, q) parameters for the GARCH model.
        steps (int): Number of days to forecast (default=21).

    Returns:
        predicted_vols (list): Predicted 21-day volatilities.
        realized_vols (list): Actual 21-day volatilities.
    """
    

    predicted_vols, realized_vols = [], []
    test_dates = df.loc[test_start:].index[:-steps]


    for day in test_dates:
        # Train ARIMA with data up to the current day
        train_series = df.loc[:day]
        arima_model = fit_arima(train_series, arima_order[0], arima_order[1], arima_order[2])
        arima_pred = forecast_arima(arima_model, steps)

        # Compute residuals and re-train GARCH model
        resid = arima_model.resid.dropna()
        garch_model = fit_garch(resid, garch_order[0], garch_order[1])
        garch_pred = forecast_garch(garch_model, steps)

        # Combine ARIMA and GARCH predictions to estimate volatility
        abs_moves = combine_arima_garch(arima_pred, garch_pred)
        vol = estimate_rolling_volatility(abs_moves)
        print(abs_moves[0:5])
        # Store predicted and realized volatilities
        predicted_vols.append(vol)
        realized_vols.append(realized_vol.loc[day])

    return predicted_vols, realized_vols




def forecast_volatility_garch(
    garch_p,
    garch_q,
    log_return_series,
    first_test_index,
    dataset,
    window_w,
    strategy
):
    """
    Prédit la volatilité en utilisant un modèle GARCH pour chaque jour de l'ensemble de test.

    Parameters:
    -----------
    garch_p : int
        Ordre p du modèle GARCH.
    garch_q : int
        Ordre q du modèle GARCH.
    log_return_series : pd.Series
        Série temporelle des log-returns, indexée par date.
    first_test_index : pd.Timestamp
        Date de début de l'ensemble de test.
    dataset : pd.DataFrame
        Doit contenir une colonne 'Log_Return' et une colonne 'Volatility_Future_{w}_days'.
    window_w : int
        Taille de la fenêtre w pour le calcul de la volatilité.

    Returns:
    --------
    df_predicted_vol : pd.DataFrame
        DataFrame indexé par date avec les volatilités prédites.
    df_realized_vol : pd.DataFrame
        DataFrame indexé par date avec les volatilités réalisées.
    """
    pred_vols = []
    realized_vols = []
    pred_dates = []

    # Extraire les dates de l'ensemble de test
    test_dates = log_return_series.index[log_return_series.index >= first_test_index]

    for date in test_dates:
        # Sélectionner les données d'entraînement jusqu'à la date actuelle
        train_data = log_return_series[:date]

        try:
            # Entraîner le modèle GARCH
            garch_model = arch_model(
                train_data,
                mean='Zero',  # Utiliser 'Zero' au lieu de 'AR'
                vol='GARCH',
                p=garch_p,
                q=garch_q,
                dist='normal'
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_garch = garch_model.fit(disp='off')

            if strategy == 'multi':
                # Prédire la volatilité pour les w prochains jours
                garch_forecast = fitted_garch.forecast(horizon=window_w)
                cond_var = garch_forecast.variance.iloc[-1].values  # Variances conditionnelles prédites
                # Calculer la volatilité prédite
                predicted_vol = np.sqrt(252 * (cond_var.sum() / (window_w -1)))
            
            if strategy == 'single':
                garch_forecast = fitted_garch.forecast(horizon=1)
                cond_var = garch_forecast.variance.values[0]
                # Calculer la volatilité prédite 
                predicted_vol = (np.sqrt(252 *cond_var))

            pred_vols.append(predicted_vol)
            pred_dates.append(date)

            # Extraire la volatilité réalisée correspondante
            realized_vol = dataset.loc[date]
            realized_vols.append(realized_vol)

        except Exception as e:
            print(f"Erreur à la date {date}: {e}")
            continue

    # Créer les DataFrames de résultats
    df_predicted_vol = pd.DataFrame(
        {'Predicted_Volatility': pred_vols},
        index=pred_dates
    )

    df_realized_vol = pd.DataFrame(
        {'Realized_Volatility': realized_vols},
        index=pred_dates
    )

    return df_predicted_vol, df_realized_vol





def plot_predicted_vs_realized_volatility(df_predicted_vol, df_realized_vol, title='Comparaison Volatilité Prédite vs Réalisée'):
    """
    Trace les volatilités prédites et réalisées sur le même graphique.

    Parameters:
    -----------
    df_predicted_vol : pd.DataFrame
        DataFrame contenant les volatilités prédites, avec une colonne 'Predicted_Volatility'.
    df_realized_vol : pd.DataFrame
        DataFrame contenant les volatilités réalisées, avec une colonne 'Realized_Volatility'.
    title : str
        Titre du graphique.
    """
    plt.figure(figsize=(14, 7))
    
    # Tracer la volatilité prédite
    plt.plot(df_predicted_vol.index, df_predicted_vol, label='Volatilité Prédite', color='blue')
    
    # Tracer la volatilité réalisée
    plt.plot(df_realized_vol.index, df_realized_vol, label='Volatilité Réalisée', color='orange')
    
    # Ajouter le titre et les labels des axes
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Volatilité')
    
    # Ajouter une légende
    plt.legend()
    
    # Afficher le graphique
    plt.show()

    # Calcul de la RMSE
    rmse = np.sqrt(mean_squared_error(df_predicted_vol.values, df_realized_vol.values))
    print(f"RMSE: {rmse}")







def plot_series(series, title):
    plt.figure(figsize=(10, 4))
    plt.plot(series)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()