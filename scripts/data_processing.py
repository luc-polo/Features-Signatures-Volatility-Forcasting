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

    Note: the Daily Returns column is no longer kept in the final dataset.

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

    # If the Daily Returns column is still present, remove it
    if "Daily Returns" in data.columns:
        data.drop(columns=["Daily Returns"], inplace=True)

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
    Normalizes selected features using scikit-learn's StandardScaler (or similar).
    Also adds a normalized time column (manual min-max scaling for the index).

    Features to be normalized with StandardScaler in this example:
      - Log Mid-Price
      - Log Return
      - Log Mid-Price Return
      - Spread
      - Imbalance
      - Volume (via cumulative volume if desired)
      - Moving Average (20 days)

    We separate at the end:
      - gold_data: raw + minimal derived metrics
      - normalized_data: only the normalized/transformed columns

    Parameters:
        data (pd.DataFrame): The dataset with metrics added.

    Returns:
        (pd.DataFrame, pd.DataFrame): (gold_data, normalized_data)
    """

    # Make a copy so as not to modify the original dataframe
    gold_data = data.copy()

    # 1. Normalize time (index) with min-max scaling to [0, 1]
    gold_data["Normalized Time"] = (
        (gold_data.index - gold_data.index[0]) / (gold_data.index[-1] - gold_data.index[0])
    )

    # 2. StandardScaler for a selected set of columns
    columns_to_scale = [
        "Log Mid-Price",
        "Log Return",
        "Log Mid-Price Return",
        "Spread",
        "Imbalance",
        "Moving Average (20 days)",
        "Volume"
    ]

    # Prepare data for fitting
    scaler = StandardScaler()

    subset_for_scaling = gold_data[columns_to_scale]

    scaled_values = scaler.fit_transform(subset_for_scaling)

    # Store scaled values back under "Normalized <column>" names
    for i, col in enumerate(columns_to_scale):
        gold_data[f"Normalized {col}"] = scaled_values[:, i]

    # Build the final DataFrame with only the normalized columns
    normalized_cols = [f"Normalized {c}" for c in columns_to_scale] + ["Normalized Time"]
    normalized_data = gold_data[normalized_cols].copy()

    # Drop them from gold_data to keep it as "unscaled" data
    gold_data.drop(columns=normalized_cols, inplace=True, errors="ignore")

    return gold_data, normalized_data


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


def compute_signature(data, order):
    """
    Compute the signature of the given data up to a specified order.

    Parameters:
        data (pd.DataFrame): Input time-series data (e.g., lead-lag transformed data).
        order (int): The order of the signature to compute.

    Returns:
        np.ndarray: The computed signature.
    """
    # Ensure data is sorted by index
    data = data.sort_index()

    # Convert the DataFrame to a NumPy array (required by `esig`)
    path = data.values

    # Compute the signature up to the specified order
    signature = ts.stream2sig(path, order)
    
    return signature
