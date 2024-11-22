from scripts.data_loader import get_gold_data
from scripts.data_processing import add_metrics
from scripts.data_visualization import plot_gold_price, plot_daily_returns

# Configuration
GOLD_SYMBOL = 'GC=F'
START_DATE = '2020-01-01'
END_DATE = '2024-10-01'

# Load or download data
gold_data = get_gold_data(GOLD_SYMBOL, START_DATE, END_DATE, refresh=False)

# Add metrics
gold_data = add_metrics(gold_data)

# Visualize data
plot_gold_price(gold_data)
plot_daily_returns(gold_data)

