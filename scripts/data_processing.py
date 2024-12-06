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
    data["Imbalance"] = data["Imbalance"].fillna(0)  # Handle missing values in imbalance

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



def apply_lead_lag(data):
    """
    Applies a lead-lag transformation to create a zigzag-like path for normalized features.
    This duplicates each row of the DataFrame into "lead" and "lag" values to simulate
    a continuous zigzag path. It also creates specific columns for visualization.
    
    Parameters:
        data (pd.DataFrame): The original data with normalized features.

    Returns:
        pd.DataFrame: A DataFrame with the lead-lag transformation applied.
    """
    # Initialize an empty DataFrame for the lead-lag path
    lead_lag_path = pd.DataFrame()

    # Create a "lagged" DataFrame by shifting the original data
    lag_data = data.shift(1)
    lag_data["Type"] = "Lag"

    # Create a "lead" DataFrame from the original data
    lead_data = data.copy()
    lead_data["Type"] = "Lead"

    # Concatenate lead and lag data alternately to create the zigzag path
    lead_lag_path = pd.concat([lag_data, lead_data]).sort_index().reset_index(drop=True)

    # Handle the first row where the lag data would have NaN values (duplicate the first lead row)
    lead_lag_path.iloc[0] = lead_data.iloc[0]

    # Explicitly create a column for "Lead-Lag Mid-Price" if "Normalized Log Mid-Price" exists
    if "Normalized Log Mid-Price" in lead_lag_path.columns:
        lead_lag_path["Lead-Lag Mid-Price"] = lead_lag_path["Normalized Log Mid-Price"]

    # Debugging information: Show the first few rows of both lead and lag sections
    print("Lead-Lag Transformation Applied: Preview of DataFrame:")
    print(lead_lag_path.groupby("Type").head(3))  # Display a preview of lead and lag rows

    return lead_lag_path
