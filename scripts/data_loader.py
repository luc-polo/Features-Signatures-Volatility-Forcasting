import yfinance as yf
import os
import pandas as pd

# Local file name to save the data
DATA_DIR = "data"
FILE_NAME = "gold_data.csv"
LOCAL_FILE = os.path.join(DATA_DIR, FILE_NAME)


# Function to download data
def download_data(symbol, start_date, end_date):
    """Downloads gold data from Yahoo Finance and saves it locally."""
    print("Downloading data from Yahoo Finance...")
    data = yf.download(symbol, start=start_date, end=end_date)
    os.makedirs(DATA_DIR, exist_ok=True)  # Ensure the data directory exists
    data.to_csv(LOCAL_FILE, index=True)  # Include the index (Date) when saving
    print(f"Data downloaded and saved to {LOCAL_FILE}.")
    return data

# Function to load data from a local file
def load_local_data():
    """Loads data from the local CSV file."""
    print(f"Loading data from {LOCAL_FILE}...")
    return pd.read_csv(LOCAL_FILE, index_col='Date', parse_dates=True)

# Check if local data exists or needs to be refreshed
def get_gold_data(symbol, start_date, end_date, refresh=False):
    """Gets gold data from the local file or downloads it if refresh is True."""
    if os.path.exists(LOCAL_FILE) and not refresh:
        return load_local_data()
    else:
        return download_data(symbol, start_date, end_date)
