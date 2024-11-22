def add_metrics(data):
    """Adds metrics such as daily returns and moving averages to the dataset."""
    data['Daily Returns'] = data['Close'].pct_change()  # Daily returns
    data['Moving Average (20 days)'] = data['Close'].rolling(window=20).mean()  # 20-day moving average
    return data
