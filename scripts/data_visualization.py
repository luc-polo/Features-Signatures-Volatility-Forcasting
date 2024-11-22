import matplotlib.pyplot as plt

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
