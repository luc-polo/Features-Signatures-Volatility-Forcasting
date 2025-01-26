import numpy as np
import pandas as pd
import esig.tosig as ts
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
import warnings

# Helper function to compute forward (future) rolling volatility
# excluding the current day. For row i, it calculates std of [i+1 ... i+w].
def compute_future_volatility(log_return_series, window):
    """
    Computes the rolling standard deviation over the *next* `window` days,
    excluding the current day. Returns a series aligned with the original index.
    """
    # Reverse the series
    reversed_series = log_return_series[::-1]
    # Shift by 1 to exclude the 'current' day in reversed coordinates
    reversed_series_shifted = reversed_series.shift(1)
    # Rolling std on the reversed, shifted series
    reversed_rolling_std = reversed_series_shifted.rolling(window=window).std()
    # Reverse back to align with the original index
    return reversed_rolling_std[::-1]

def add_metrics(data):
    """
    Adds derived metrics to the dataset:
      - Log Price: Logarithm of the closing price.
      - Log Return: Daily log return calculated as the difference of Log Price.
      - Moving Average (20 days): 20-day rolling mean of the closing price.
      - Log Mid-Price: Logarithm of the average of High and Low prices.
      - Log Mid-Price Return: Daily log return of the Log Mid-Price.
      - Spread: Difference between High and Low prices.
      - Imbalance: Relative volume difference between consecutive days.
      - Volatility (Past and Future): Annualized volatility over specified window sizes for both past and future days.
    
    Parameters:
        data (pd.DataFrame): The original dataset with columns ['High', 'Low', 'Close', 'Volume'].
    
    Returns:
        pd.DataFrame: The dataset with the new metrics added.
    """
    # Log Price
    data["Log Price"] = np.log(data["Close"])

    # Log Return (daily difference of Log Price)
    data["Log Return"] = data["Log Price"] - data["Log Price"].shift(1)

    # Moving Average (20-day rolling mean of the closing price)
    data["Moving Average (20 days)"] = data["Close"].rolling(window=20).mean()

    # Moving Average (10-day rolling mean of the closing price)
    data["Moving Average (10 days)"] = data["Close"].rolling(window=10).mean()

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
    window_sizes = [3, 10, 21, 50, 260]
    for w in window_sizes:
        # Past Volatility
        past_vol_col = f"Volatility_Past_{w}_days"
        data[past_vol_col] = np.sqrt(252) * data["Log Return"].rolling(window=w).std()

        # Future Volatility (e.g. Volatility_Future_10_days), excluding current day
        future_vol_col = f"Volatility_Future_{w}_days"
        data[future_vol_col] = (
            np.sqrt(252) * compute_future_volatility(data["Log Return"], w))
    
    return data

def clean_dataframes(*dataframes):
    """
    Remove rows with NaN values across multiple DataFrames and print the number of removed rows.

    Parameters:
        *dataframes: Variable number of pandas DataFrames.

    Returns:
        Tuple of cleaned DataFrames, in the same order as provided.
    """
    # Combine all indices with NaNs across DataFrames
    all_nan_indices = set().union(*[df[df.isna().any(axis=1)].index for df in dataframes])

    # Print the number of rows removed
    print(f"Removed {len(all_nan_indices)} rows with missing values.")

    # Drop rows with these indices from all DataFrames
    cleaned_dataframes = tuple(df.drop(index=all_nan_indices) for df in dataframes)

    # Print the number of rows remaining for each DataFrame
    for i, df in enumerate(cleaned_dataframes, start=1):
        print(f"DataFrame {i}: {len(df)} rows remaining.")

    return cleaned_dataframes



def remove_missing_rows(df: pd.DataFrame):
    """
    Removes all rows with any missing values from the DataFrame and displays
    the number of rows removed, their original indices, and how many rows remain.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The DataFrame with rows containing NaNs removed.
    """
    missing_indices = df[df.isna().any(axis=1)].index
    print(f"Removed {len(missing_indices)} rows with missing values. Indices: {list(missing_indices)}")
    df_cleaned = df.dropna()
    print(f"Remaining rows in the DataFrame: {len(df_cleaned)}")
    return df_cleaned

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



def revert_lead_lag(dataset):
    """
    Reverts a Lead-Lag transformed dataset back to its original structure 
    by keeping only the rows with odd line numbers (starting from 0).
    The original indices are preserved.

    Parameters:
        dataset (pd.DataFrame): The Lead-Lag transformed dataset.

    Returns:
        pd.DataFrame: A dataset containing only the original rows from 
                      the Lead-Lag transformation, with their original indices.
    """
    # Select rows with odd line numbers (1-based odd is 0-based even index)
    odd_rows = dataset.iloc[::2]
    
    return odd_rows





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
        col_names = [f"sig_w{w}_ord{key.count(',') + 1}_{key}" for key in signature_keys]
        
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

def compute_signature_differences(signature_df, w):
    """
    Calculate the differences between second-order signature terms for all unique pairs of variables.

    Parameters
    ----------
    signature_df : pd.DataFrame
        DataFrame containing the second-order signature features (output of `compute_signature`).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the calculated differences for all unique pairs of variables, 
        with appropriately named columns.
    """
    # Extraire les colonnes de signature d'ordre deux
    second_order_cols = [col for col in signature_df.columns if col.startswith("sig_") and col.count(',') == 1]
    
    # Identifier les paires uniques de variables (x, y) où x < y
    variable_pairs = set()
    for col in second_order_cols:
        try:
            indices = col.split('(')[-1].split(')')[0]  # Extraire la partie "x,y"
            x, y = map(int, indices.split(','))
            if x < y:  # Assurer que chaque paire est unique et que x < y
                variable_pairs.add((x, y))
        except (ValueError, IndexError):
            # Ignorer les colonnes qui ne suivent pas le format attendu
            continue
    
    # Calculer les différences pour chaque paire unique
    differences_data = {
        f"diff_{x}_{y}": signature_df[f"sig_w{w}_ord2_({x},{y})"] - signature_df[f"sig_w{w}_ord2_({y},{x})"]
        for x, y in variable_pairs 
        if f"sig_w{w}_ord2_({x},{y})" in signature_df.columns and f"sig_w{w}_ord2_({y},{x})" in signature_df.columns
    }
    
    # Convertir le dictionnaire en DataFrame
    differences_df = pd.DataFrame(differences_data, index=signature_df.index)

    return differences_df






def split_train_test(original_df: pd.DataFrame, 
                    signatures_df: pd.DataFrame, 
                    target_df: pd.DataFrame, 
                    target_name: str, 
                    test_size: float = 0.2, 
                    random_state: int = 42):
    """
    Splits the data into training and testing sets based on original features, signatures, and target.

    Parameters:
        original_df (pd.DataFrame): DataFrame containing original features (excluding the target).
        signatures_df (pd.DataFrame): DataFrame containing signature features.
        target_df (pd.DataFrame): DataFrame containing the target feature(s).
        target_name (str): Name of the target column in target_df.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int, optional): Random state for reproducibility. Default is 42.

    Returns:
        tuple: A tuple containing:
            - X_train (pd.DataFrame): Training predictors.
            - X_test (pd.DataFrame): Testing predictors.
            - Y_train (pd.Series): Training target.
            - Y_test (pd.Series): Testing target.
    """
    # Step 1: Identify indices without any NaNs in signatures_df
    valid_indices = signatures_df.dropna().index

    # Step 2: Align original_df and target_df to these valid_indices
    original_aligned = original_df.loc[valid_indices]
    signatures_aligned = signatures_df.loc[valid_indices]
    target_aligned = target_df.loc[valid_indices, target_name]

    # Step 3: Combine original features and signatures
    X_combined = pd.concat([original_aligned, signatures_aligned], axis=1)

    # Step 4: Split into training and testing sets
    # Determine split index based on test_size
    split_index = int(len(X_combined) * (1 - test_size))

    # Split the data
    X_train = X_combined.iloc[:split_index]
    X_test = X_combined.iloc[split_index:]
    Y_train = target_aligned.iloc[:split_index]
    Y_test = target_aligned.iloc[split_index:]


    return X_train, X_test, Y_train, Y_test


def split_time_series(df, target_column, test_proportion):
    """
    Splits a time series dataframe into training and test sets in chronological order, 
    based on the specified proportion of samples in the test set. Returns additional 
    information for convenience.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the time series data.
    target_column : str
        The name of the column to be used as the target feature.
    test_proportion : float
        The proportion of samples to include in the test set (e.g., 0.2 for 20%).

    Returns:
    --------
    series_train : pandas.Series
        The training portion of the target column.
    series_test : pandas.Series
        The test portion of the target column.
    first_test_index : pandas.Timestamp or int
        The index of the first test set entry.
    target_series : pandas.Series
        The complete target series extracted from the dataframe.
    """
    target_series = df[target_column]
    split_index = int(len(target_series) * (1 - test_proportion))
    series_train = target_series.iloc[:split_index]
    series_test = target_series.iloc[split_index:]
    first_test_index = series_test.index[0]
    return series_train, series_test, first_test_index, target_series

def rank_predictors_by_correlation(target_df, *predictor_dfs):
    """
    Ranks predictors from multiple DataFrames by the absolute value of their correlation with the target feature.

    Parameters:
        target_df: DataFrame containing the target feature.
        *predictor_dfs: Any number of DataFrames containing predictor features.

    Returns:
        A sorted DataFrame of predictors and their absolute correlation coefficients.
    """
    # Combine all predictors into one DataFrame
    predictors = pd.concat(predictor_dfs, axis=1)

    # Compute correlations with the target feature
    correlations = predictors.corrwith(target_df.squeeze()).abs().sort_values(ascending=False)

    # Return the sorted correlations as a DataFrame
    return correlations.reset_index(name='Absolute Correlation').rename(columns={'index': 'Predictor'})



def filter_predictors_by_correlation(df1, df2, target_df, threshold):
    """
    Filters predictor DataFrames by removing columns with correlation below the threshold.

    Parameters:
        df1, df2, df3: Predictor DataFrames.
        target_df: DataFrame containing the target feature.
        threshold: Correlation threshold.

    Returns:
        Filtered predictor DataFrames.
    """
    def filter_df(df):
        correlations = df.corrwith(target_df.squeeze()).abs()
        removed_features = len(correlations[correlations < threshold])
        print(f"Removed {removed_features} features from a DataFrame.")
        return df.loc[:, correlations >= threshold]

    return filter_df(df1), filter_df(df2)


import pandas as pd

def get_highly_correlated_features(predictors_df, target, threshold):
    """
    Selects predictor features that have an absolute correlation with the target above the specified threshold.

    Parameters:
        predictors_df (pd.DataFrame): DataFrame containing predictor features.
        target (pd.Series or pd.DataFrame): Series or single-column DataFrame containing the target feature.
        threshold (float): Minimum absolute correlation required to retain a feature.

    Returns:
        pd.DataFrame: DataFrame containing only the highly correlated features.
    """
    # Ensure target is a Series
    if isinstance(target, pd.DataFrame):
        if target.shape[1] != 1:
            raise ValueError("target DataFrame must have exactly one column.")
        target = target.iloc[:, 0]
    elif not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series or single-column DataFrame.")
    
    # Calculate correlations
    correlations = predictors_df.corrwith(target).abs()
    
    # Filter features based on threshold
    selected_features = correlations[correlations >= threshold].index
    filtered_df = predictors_df[selected_features]
    
    # Display the number of retained features
    print(f"Number of features retained: {len(selected_features)}")
    
    return filtered_df



def get_highly_correlated_quadratics(predictors_df, target_df, threshold):
    """
    Identifies quadratic terms and their combinations that are most correlated with the target.

    Parameters:
        predictors_df: DataFrame containing the predictor features.
        target_df: DataFrame containing the target feature.
        threshold: Correlation threshold.

    Returns:
        A DataFrame of filtered quadratic terms and their correlations with the target.
    """
    
    # Initialize PolynomialFeatures to generate quadratic terms
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    
    # Generate quadratic terms
    quadratic_data = poly.fit_transform(predictors_df)
    feature_names = poly.get_feature_names_out(predictors_df.columns)
    
    # Create a DataFrame for quadratic terms
    quadratic_df = pd.DataFrame(quadratic_data, columns=feature_names, index=predictors_df.index)

    # Compute correlations with the target
    correlations = quadratic_df.corrwith(target_df.squeeze())
 
    # Filter by the threshold
    filtered_terms = correlations[correlations.abs() >= threshold]
 
    # Select only the columns corresponding to the filtered terms
    filtered_quadratic_df = quadratic_df[filtered_terms.index]
    
    # Display the number of retained variables
    print(f"Number of quadratic terms retained : {len(filtered_terms)}")
    
    return filtered_quadratic_df





