import yfinance as yf
import os
import pandas as pd

DATA_DIR = os.path.join(os.getcwd(), "data")  # Ensure the data directory is inside the current working directory
FILE_NAME = "gold_data.csv"
LOCAL_FILE = os.path.join(DATA_DIR, FILE_NAME)

def download_data(symbol, start_date, end_date):
    """Download data from Yahoo Finance, process it, and save it locally."""
    print("Downloading data from Yahoo Finance...")
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Flatten column levels if they are multi-indexed
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
    
    # Rename columns for consistency
    data.columns = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    
    # Reset index to include "Date" as a regular column
    data.reset_index(inplace=True)

    # Ensure the data directory exists and save the file
    os.makedirs(DATA_DIR, exist_ok=True)
    data.to_csv(LOCAL_FILE, index=False)
    print(f"Data saved to: {LOCAL_FILE}")
    return data.set_index("Date")


def load_local_data():
    """Load data from the local CSV file and ensure proper formatting."""
    print(f"Loading data from {LOCAL_FILE}...")
    data = pd.read_csv(LOCAL_FILE, parse_dates=["Date"]).set_index("Date")
    print("Loaded DataFrame columns:")
    print(data.columns)
    return data



# Check if local data exists or needs to be refreshed
def get_gold_data(symbol, start_date, end_date, refresh=False):
    """Gets gold data from the local file or downloads it if refresh is True."""
    # Resolve the absolute path of the file
    full_path = LOCAL_FILE
    
    # Check if the file exists and refresh is not requested
    if os.path.exists(full_path) and not refresh:
        print(f"File found at: {full_path}. \n Loading data...\n")
        return load_local_data()
    else:
        print(f"File not found or refresh requested. Downloading new data...")
        return download_data(symbol, start_date, end_date)
