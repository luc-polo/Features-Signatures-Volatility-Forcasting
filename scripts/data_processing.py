import numpy as np
import pandas as pd
import esig.tosig as ts
from sklearn.preprocessing import StandardScaler

def add_metrics(data):
    """
    Adds derived metrics to the dataset:
      - Log Price: log of the closing price
      - Log Return: difference of Log Price (daily log return)
      - Moving Average (20 days)
      - Log Mid-Price: log of the average (High + Low)/2
      - Log Mid-Price Return: difference of Log Mid-Price (log return)
      - Spread: difference between High and Low
      - Imbalance: relative volume difference between two consecutive days
      - Volatility: annualized volatility over specified window sizes

    Parameters:
        data (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The dataset with the new metrics added.
    """
    # Log Price
    data["Log Price"] = np.log(data["Close"])

    # Log Return (daily difference of Log Price)
    data["Log Return"] = data["Log Price"] - data["Log Price"].shift(1)

    # Moving Average (20-day rolling mean of the closing price)
    data["Moving Average (20 days)"] = data["Close"].rolling(window=20).mean()

    # Log Mid-Price
    data["Log Mid-Price"] = np.log(0.5 * (data["High"] + data["Low"]))

    # Log Mid-Price Return
    data["Log Mid-Price Return"] = data["Log Mid-Price"] - data["Log Mid-Price"].shift(1)

    # Spread (High - Low)
    data["Spread"] = data["High"] - data["Low"]

    # Imbalance: (Volume(t) - Volume(t-1)) / (Volume(t) + Volume(t-1))
    data["Imbalance"] = (data["Volume"] - data["Volume"].shift(1)) / (
        data["Volume"] + data["Volume"].shift(1)
    )

    # Volatility calculations for specified window sizes
    window_sizes = [10, 21, 50, 260]
    for w in window_sizes:
        volatility_col = f"Volatility_{w}_days"
        # Calculate rolling standard deviation of Log Return
        data[volatility_col] = (
            np.sqrt(252) * data["Log Return"].rolling(window=w).std()
        )

    return data


def missing_values_checking(data):
    """Check for missing values in our dataset."""
    if data.isnull().any().any():
        print("Warning: Missing values detected in the data.")
        print(data.isnull().sum())  # Show count of missing values per column
    else:
        print("No missing values detected.")


def normalize_features(data):
    """
    Normalizes the time (index) column using min-max scaling to [0, 1].

    We separate at the end:
      - gold_data: raw + minimal derived metrics
      - normalized_data: only the normalized time column

    Parameters:
        data (pd.DataFrame): The dataset with metrics added.

    Returns:
        (pd.DataFrame, pd.DataFrame): (gold_data, normalized_data)
    """

    # Make a copy so as not to modify the original dataframe
    gold_data = data.copy()

    # 1. Normalize time (index) with min-max scaling to [0, 1]
    normalized_time = (gold_data.index - gold_data.index[0]) / (gold_data.index[-1] - gold_data.index[0])
    gold_data["Normalized Time"] = normalized_time

    return gold_data


def apply_lead_lag(data, lead_lag_columns=None):
    """
    Applies a Lead-Lag transformation to specified columns of a DataFrame.

    For each selected column, adds:
        - `<column>_Lag`: The value from the previous time step.
        - `<column>_Lead`: The value from the current or next time step.

    Parameters:
        data (pd.DataFrame): The input time-series data.
        lead_lag_columns (list, optional): Columns to apply the transformation. 
                                           Defaults to all columns.

    Returns:
        pd.DataFrame: A DataFrame with added `_Lag` and `_Lead` columns for specified variables.
    """
    # Sort the data by its index
    data = data.sort_index()
    
    # If no columns specified, apply the transformation to all columns
    if lead_lag_columns is None:
        lead_lag_columns = data.columns.tolist()
        
    # Columns for Lead-Lag transformation
    ll = [c for c in lead_lag_columns if c in data.columns]
    # Other columns not included in Lead-Lag transformation
    lo = [c for c in data.columns if c not in ll]

    # Return an empty DataFrame if input is empty
    if data.empty:
        return pd.DataFrame()

    rows = []
    
    # First row initialization (t_1)
    first = data.iloc[0]
    row_init = {}
    for c in ll:
        row_init[c + "_Lag"], row_init[c + "_Lead"] = first[c], first[c]
    for c in lo:
        row_init[c + "_Lead"] = first[c]
    rows.append((data.index[0], row_init))

    # For each interval [t_{i-1}, t_i], add two rows
    for i in range(1, len(data)):
        prev, curr = data.iloc[i-1], data.iloc[i]
        
        # Row A: Values from [t_{i-1}, t_i]
        rowA = {}
        for c in ll:
            rowA[c + "_Lag"], rowA[c + "_Lead"] = prev[c], curr[c]
        for c in lo:
            rowA[c + "_Lead"] = curr[c]
        rows.append((data.index[i], rowA))

        # Row B: Values from [t_i, t_i]
        rowB = {}
        for c in ll:
            rowB[c + "_Lag"], rowB[c + "_Lead"] = curr[c], curr[c]
        for c in lo:
            rowB[c + "_Lead"] = curr[c]
        rows.append((data.index[i], rowB))

    # Build the final DataFrame
    return pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows])


def compute_signature(df,
                      order=2,
                      windows=[10],
                      exclude_cols=None):
    """
    Calculates the path signature of specified order over multiple rolling windows 
    and returns them as a separate DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the daily data.
    order : int, optional
        The order of the signature to calculate. Default is 2.
    windows : list of int, optional
        List of window sizes (in days) to use for rolling signature calculations. 
        Default is [5, 10].
    exclude_cols : list of str, optional
        List of column names to exclude from the signature calculation. 
        Default is None (no exclusions).
        
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated signature features, with 
        column names indicating the window size and signature components.
    """
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Select numeric columns and exclude specified columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    signature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Dictionary to hold the computed signature columns
    signature_data = {}
    
    for w in windows:
        d = len(signature_cols)
        
        # Récupération des "clefs" de la signature et conversion en liste
        # ts.sigkeys(d, order) -> par exemple "() (1) (2) (1,2)" qu'on split
        keys_str = ts.sigkeys(d, order)
        signature_keys = keys_str.split()  # Liste de chaînes, ex : ["()", "(1)", "(2)", "(1,2)", ...]
        signature_keys = [key for key in signature_keys if key != "()"]

        # Création des noms de colonnes explicites
        #   Ex: sig_w5_ord2_()  sig_w5_ord2_(1)  sig_w5_ord2_(2)  sig_w5_ord2_(1,2) ...
        col_names = [f"sig_w{w}_ord{order}_{key}" for key in signature_keys]
        
        # Initialiser les colonnes avec NaN
        for cn in col_names:
            signature_data[cn] = [np.nan] * len(df)
        
        # Calcul des signatures sur fenêtres glissantes
        for i in range(len(df)):
            start_idx = i - w + 1
            if start_idx < 0:
                # Pas assez de points pour former une fenêtre de taille w
                continue
            
            # Fenêtre courante sur les colonnes à inclure dans la signature
            window_df = df.iloc[start_idx : i + 1][signature_cols]
            # Conversion en array
            path = window_df.to_numpy()
            
            # Calcul de la signature d'ordre spécifié
            sig_values = ts.stream2sig(path, order)
            
            # Assignation des valeurs aux colonnes
            for cn, val in zip(col_names, sig_values[1:]):
                signature_data[cn][i] = val
    
    # Assemblage final en un DataFrame
    signatures_df = pd.DataFrame(signature_data, index=df.index)

    return signatures_df