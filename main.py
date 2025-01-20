import scripts.data_loader as data_loader
import scripts.data_processing as data_processing
import scripts.data_visualization as data_visualization

# Increase the number of columns displayed by pandas
import pandas as pd
pd.set_option('display.max_columns', None)

# Configuration
GOLD_SYMBOL = 'GC=F'
START_DATE = '2022-01-01'
END_DATE = '2024-10-01'

# Load or download data
gold_data = data_loader.get_gold_data(GOLD_SYMBOL, START_DATE, END_DATE, refresh=False)

# Check for missing values
data_processing.missing_values_checking(gold_data)


# - **Log Price**: Logarithm of the "Close" price, providing a scaled and stabilized version of the closing price.
# - **Log Return**: Daily difference of Log Price, capturing daily price momentum in a log-scaled format.
# - **Moving Average (20 days)**: Rolling average of the "Close" price over 20 days, highlighting longer-term trends.
# - **Log Mid-Price**: Logarithm of the mid-price (average of "High" and "Low"), stabilizing variance and converting
#   multiplicative changes into additive ones for better analysis.
# - **Log Mid-Price Return**: Daily difference of Log Mid-Price, capturing log-scaled variations in mid-price.
# - **Spread**: Difference between "High" and "Low" prices, indicating price volatility within a time interval.
# - **Imbalance**: Relative volume difference between consecutive intervals, capturing market sentiment and activity.
gold_data = data_processing.add_metrics(gold_data)

# Visualize all key metrics in a single multi-panel plot
data_visualization.plot_metrics(gold_data)

# Outliers identification
# Plot box plots for all numeric variables
data_visualization.plot_boxplots(gold_data, plots_per_column=8)

# Step 3: Normalize the following features:
# Time, Log Mid-Price, Spread, Imbalance, Cumulative Volume, Daily Returns, and Moving Average (20 days).
gold_data, normalized_data = data_processing.normalize_features(gold_data)

# Print column names of gold_data and normalized_data for verification
print("Columns in gold_data:", gold_data.columns)
print("Columns in normalized_data:", normalized_data.columns)

# Step 4: Apply lead-lag transformation
lead_lag_data = data_processing.apply_lead_lag(normalized_data, lead_lag_columns=['Normalized Log Mid-Price'])

# Visualize the transformed features lead-lag
data_visualization.plot_lead_lag(lead_lag_data, ["Normalized Log Mid-Price"])

# Compute the signature of order 3 for the lead-lag data
signature_order_3 = data_processing.compute_signature(
    lead_lag_data[['Normalized Log Mid-Price_Lag', 'Normalized Log Mid-Price_Lead']], order=3
)

# Print the resulting signature
print("Signature of order 3:")
print(signature_order_3)
