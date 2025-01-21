import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd



def plot_metrics(data):
    """
    Traces toutes les métriques numériques présentes dans le DataFrame sous forme de graphiques multi-panneaux 
    avec les graduations des dates sur chaque axe des abscisses.
    
    Parameters:
        data (pd.DataFrame): Le DataFrame contenant les métriques à tracer.
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        raise ValueError("Aucune colonne numérique trouvée dans le DataFrame pour le tracé.")
    
    # Définir la disposition des sous-graphiques (2 colonnes par défaut)
    cols = 2
    rows = math.ceil(len(numeric_cols) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharex=False)  # sharex=False pour avoir des axes x indépendants
    axes = axes.flatten() if rows > 1 else [axes]
    
    # Tracer chaque colonne numérique avec les graduations des dates sur l'axe des abscisses
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.plot(data.index, data[col], label=col, alpha=0.8)
        ax.set_title(col, fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylabel(col)
        ax.set_xlabel("Date")  # Ajouter "Date" sur tous les axes des abscisses
        ax.tick_params(axis='x', rotation=45)  # Rotation des labels des dates pour une meilleure lisibilité
    
    # Supprimer les axes inutilisés si le nombre de graphiques n'est pas un multiple de cols
    for j in range(len(numeric_cols), len(axes)):
        fig.delaxes(axes[j])
    
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


def plot_boxplots(data, columns=None, plots_per_column=3):
    """
    Plots box plots for multiple variables, arranged vertically with automatic layout adjustment.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (list or None): The columns to plot. If None, all numeric columns are plotted.
        plots_per_column (int): Number of plots per column in the layout.
    """
    if columns is None:
        columns = data.select_dtypes(include="number").columns.tolist()
    
    if not columns:
        raise ValueError("No numeric columns found to plot.")
    
    num_columns = len(columns)
    num_rows = math.ceil(num_columns / plots_per_column)
    
    fig, axes = plt.subplots(num_rows, plots_per_column, figsize=(6 * plots_per_column, 5 * num_rows))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        ax = axes[i]
        ax.boxplot(data[column].dropna(), vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightblue", alpha=0.7))
        ax.set_title(column, fontsize=12)
        ax.axhline(np.median(data[column].dropna()), color="orange", label="Median", linestyle="-")
        ax.axhline(data[column].mean(), color="green", label="Mean", linestyle="--")
        ax.legend(loc="upper right")
        ax.set_ylabel("Values")
    
    for j in range(num_columns, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_lead_lag(data, variables, start=0, end=20):
    """
    Plot lead and lag data for specified variables over a specified range. Just for checking it's well computed.
    
    Parameters:
    - data: DataFrame returned by apply_lead_lag.
    - variables: List of variable names (without '_Lead' or '_Lag').
    - start: Start index for the interval to plot (default is 0).
    - end: End index for the interval to plot (default is 20).
    """
    # Slice the data to the desired interval
    sliced_data = data.iloc[start:end]

    sliced_data = sliced_data.reset_index(drop=True)

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
