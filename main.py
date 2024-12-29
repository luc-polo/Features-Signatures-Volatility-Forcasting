from scripts.data_loader import get_gold_data
from scripts.data_processing import add_metrics, missing_values_checking, normalize_features, apply_lead_lag
from scripts.data_visualization import plot_gold_price, plot_daily_returns, plot_boxplots, plot_lead_lag, plot_metrics

# Increase the number of columns displayed by pandas
import pandas as pd
pd.set_option('display.max_columns', None)

# Configuration
GOLD_SYMBOL = 'GC=F'
START_DATE = '2022-01-01'
END_DATE = '2024-10-01'


# Load or download data
gold_data = get_gold_data(GOLD_SYMBOL, START_DATE, END_DATE, refresh=False)

# Check for missing values
missing_values_checking(gold_data)

# - **Daily Returns**: Percentage change in the "Close" price, capturing short-term price momentum.
# - **Moving Average (20 days)**: Rolling average of the "Close" price over 20 days, highlighting longer-term trends.
# - **Log Mid-Price**: Logarithm of the mid-price (average of "High" and "Low"), stabilizing variance and converting
#   multiplicative changes into additive ones for better analysis.
# - **Spread**: Difference between "High" and "Low" prices, indicating price volatility within a time interval.
# - **Imbalance**: Relative volume difference between consecutive intervals, capturing market sentiment and activity.
gold_data = add_metrics(gold_data)

# Visualize all key metrics in a single multi-panel plot
plot_metrics(gold_data)


# Outliers identification
# Plot box plots for all numeric variables
plot_boxplots(gold_data, plots_per_column=8)


# Step 3: Normalize the following features:
# Time, Log Mid-Price, Spread, Imbalance, Cumulative Volume, Daily Returns, and Moving Average (20 days).
gold_data, normalized_data = normalize_features(gold_data)


# Print column names of gold_data and normalized_data for verification
print("Columns in gold_data:", gold_data.columns)
print("Columns in normalized_data:", normalized_data.columns)

# Step 4: Apply lead-lag transformation
lead_lag_data = apply_lead_lag(normalized_data, lead_lag_columns=['Normalized Log Mid-Price'])

# Visualize the transformed features lead-lag
plot_lead_lag(lead_lag_data,["Normalized Log Mid-Price"])

# Compute the signature of order 3 for the lead-lag data
signature_order_3 = compute_signature(lead_lag_data[['Normalized Log Mid-Price_Lag', 'Normalized Log Mid-Price_Lead']], order=3)

# Print the resulting signature
print("Signature of order 3:")
print(signature_order_3)


