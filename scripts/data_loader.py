import yfinance as yf
import os
import pandas as pd

DATA_DIR = os.path.join(os.getcwd(), "data")  # Ensure the data directory is inside the current working directory
FILE_NAME = "gold_data.csv"
LOCAL_FILE = os.path.join(DATA_DIR, FILE_NAME)


def download_data(symbol, start_date, end_date):
    """Downloads gold data from Yahoo Finance and saves it locally."""
    print("Downloading data from Yahoo Finance...")
    data = yf.download(symbol, start=start_date, end=end_date)
    
    # Assurez-vous que l'index est bien la colonne des dates
    data.reset_index(inplace=True)  # Déplace les dates dans une colonne
    data['Date'] = pd.to_datetime(data['Date'])  # Convertit en format date
    data.set_index('Date', inplace=True)  # Définit 'Date' comme index
    
    # Supprime les doublons dans les colonnes
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    0
    # Sauvegarde sans métadonnées inutiles
    data.to_csv(LOCAL_FILE, index=True)  # Sauvegarde avec l'index (dates)
    print(f"Data successfully downloaded and saved to: {LOCAL_FILE}")
    print(data)
    return data


def load_local_data():
    """Loads data from the local CSV file."""
    print(f"Loading data from {LOCAL_FILE}...\n")
    data = pd.read_csv(LOCAL_FILE, index_col='Date', parse_dates=True, skiprows=2, header=0)
    print(data)
    # Vérifie les colonnes chargées
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
