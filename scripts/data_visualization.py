import matplotlib.pyplot as plt
import numpy as np
import math

def plot_metrics(data):
    """
    Plots multiple metrics (e.g., price, moving average, daily returns) in a single multi-panel figure.
    
    Parameters:
        data (pd.DataFrame): The dataset containing the metrics to plot.
    """
    # Define the metrics to plot
    metrics = {
        "Gold Price (Close)": "Close",
        "20-Day Moving Average": "Moving Average (20 days)",
        "Daily Returns (%)": "Daily Returns",
        "Log Mid-Price": "Log Mid-Price",
        "Spread": "Spread",
        "Imbalance": "Imbalance",
    }

    # Set up the multi-panel figure
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics), sharex=True)

    # Loop through each metric and plot
    for i, (title, column) in enumerate(metrics.items()):
        ax = axes[i]
        ax.plot(data.index, data[column], label=title, alpha=0.8)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)
        if i == num_metrics - 1:  # Add x-label to the bottom-most plot
            ax.set_xlabel("Date")
        ax.set_ylabel(column)
    
    # Adjust layout for readability
    plt.tight_layout()
    plt.show()

def plot_gold_price(data):
    """Plots the gold price with a 20-day moving average."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Gold Price')
    plt.plot(data['Moving Average (20 days)'], label='20-day Moving Average')
    plt.title('Gold Price with Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.show()

def plot_daily_returns(data):
    """Plots the daily returns of gold."""
    plt.figure(figsize=(10, 6))
    plt.plot(data['Daily Returns'], label='Daily Returns', alpha=0.7)
    plt.title('Gold Price Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.show()


def plot_boxplots(data, columns=None, plots_per_column=3):
    """
    Plots box plots for multiple variables, arranged vertically with automatic layout adjustment.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (list or None): The columns to plot. If None, all numeric columns are plotted.
        plots_per_column (int): Number of plots per column in the layout.
    """
    if columns is None:
        # Automatically select numeric columns if not specified
        columns = data.select_dtypes(include='number').columns.tolist()
    
    if not columns:
        raise ValueError("No numeric columns found to plot.")
    
    num_columns = len(columns)
    num_rows = math.ceil(num_columns / plots_per_column)  # Calculate the required number of rows
    
    # Create the figure and axes
    fig, axes = plt.subplots(num_rows, plots_per_column, figsize=(6 * plots_per_column, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array to handle it easily

    for i, column in enumerate(columns):
        ax = axes[i]
        ax.boxplot(data[column].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_title(column, fontsize=12)
        ax.axhline(np.median(data[column].dropna()), color='orange', label='Median', linestyle='-')
        ax.axhline(data[column].mean(), color='green', label='Mean', linestyle='--')
        ax.legend(loc='upper right')
        ax.set_ylabel("Values")
    
    # Hide unused axes
    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout for readability
    plt.tight_layout()
    plt.show()


def plot_lead_lag(data, variables, start=0, end=20):
    """
    Plot lead and lag data for specified variables over a specified range.
    
    Parameters:
    - data: DataFrame returned by apply_lead_lag.
    - variables: List of variable names (without '_Lead' or '_Lag').
    - start: Start index for the interval to plot (default is 0).
    - end: End index for the interval to plot (default is 20).
    """
    # Slice the data to the desired interval
    sliced_data = data.iloc[start:end]

    for var in variables:
        plt.figure(figsize=(8, 4))
        plt.plot(sliced_data.index, sliced_data[f"{var}_Lag"], label=f"{var} Lag", linestyle="--")
        plt.plot(sliced_data.index, sliced_data[f"{var}_Lead"], label=f"{var} Lead", linestyle="-")
        plt.title(f"Lead and Lag for {var} (from {start} to {end})")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
