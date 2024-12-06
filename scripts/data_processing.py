import numpy as np
import pandas as pd


def add_metrics(data):
    """
    Adds derived metrics to the dataset, such as daily returns, moving averages,
    log mid-price, spread, and imbalance.
    
    Parameters:
        data (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The dataset with added metrics.
    """
    # Add daily returns
    data["Daily Returns"] = data["Close"].pct_change()

    # Add moving average (20-day rolling mean of the close price)
    data["Moving Average (20 days)"] = data["Close"].rolling(window=20).mean()

    # Add log mid-price (logarithm of the average of High and Low prices)
    data["Log Mid-Price"] = np.log(0.5 * (data["High"] + data["Low"]))

    # Add spread (difference between High and Low prices)
    data["Spread"] = data["High"] - data["Low"]

    # Add imbalance (relative difference in volume between consecutive time points)
    data["Imbalance"] = (data["Volume"] - data["Volume"].shift(1)) / (
        data["Volume"] + data["Volume"].shift(1)
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
    Normalizes the dataset features, ensuring all metrics are scaled appropriately.
    This includes transformations for time, log mid-price, spread, imbalance, and other features.

    Parameters:
        data (pd.DataFrame): The dataset with raw and derived metrics.

    Returns:
        - gold_data: A DataFrame containing only raw and minimally derived metrics.
        - normalized_data: A DataFrame containing only normalized and transformed features.
    """
    # Start by making a copy of the input data
    gold_data = data.copy()

    # Normalize time (scaled to [0, 1])
    gold_data["Normalized Time"] = (gold_data.index - gold_data.index[0]) / (
        gold_data.index[-1] - gold_data.index[0]
    )

    # Normalize log mid-price
    gold_data["Normalized Log Mid-Price"] = (
        gold_data["Log Mid-Price"] - np.mean(gold_data["Log Mid-Price"])
    ) / np.std(gold_data["Log Mid-Price"])

    # Normalize spread
    gold_data["Normalized Spread"] = (
        gold_data["Spread"] - np.mean(gold_data["Spread"])
    ) / np.std(gold_data["Spread"])

    # Normalize imbalance
    gold_data["Normalized Imbalance"] = (
        gold_data["Imbalance"] - np.mean(gold_data["Imbalance"])
    ) / np.std(gold_data["Imbalance"])

    # Normalize cumulative volume
    gold_data["Normalized Volume"] = (
        gold_data["Volume"].cumsum() / gold_data["Volume"].cumsum().iloc[-1]
    )

    # Normalize daily returns
    gold_data["Normalized Daily Returns"] = (
        gold_data["Daily Returns"] - gold_data["Daily Returns"].mean()
    ) / gold_data["Daily Returns"].std()

    # Normalize moving average (20 days)
    gold_data["Normalized Moving Average"] = (
        gold_data["Moving Average (20 days)"]
        - gold_data["Moving Average (20 days)"].mean()
    ) / gold_data["Moving Average (20 days)"].std()

    # Create a new DataFrame with only normalized and transformed features
    normalized_data = gold_data[
        [
            "Normalized Time",
            "Normalized Log Mid-Price",
            "Normalized Spread",
            "Normalized Imbalance",
            "Normalized Volume",
            "Normalized Daily Returns",
            "Normalized Moving Average",
        ]
    ].copy()

    # Drop normalized and transformed features from gold_data to keep it raw
    gold_data = gold_data.drop(columns=normalized_data.columns)

    return gold_data, normalized_data



def apply_lead_lag(data, lead_lag_columns=None):
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


