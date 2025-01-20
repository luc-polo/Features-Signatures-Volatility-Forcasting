import yfinance as yf
import os



GOLD_SYMBOL = 'GC=F'
START_DATE = '2023-10-05'
END_DATE = '2024-10-01'

data = yf.download(GOLD_SYMBOL, start=START_DATE, end=END_DATE, interval = "1h",)

print(data)