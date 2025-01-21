import yfinance as yf
import os
import pandas as pd

s = pd.Series([10, 20, 30, 40, 50, 60])

# Fenêtre roulante sur les 2 prochaines valeurs pour la colonne 'A'
#df['Future Rolling Mean'] = df.shift(-1).rolling(window=2).mean()


print(s)
print(s.shift(-1).rolling(window=2).mean())
print(s.shift(-2).rolling(window=2).mean())
print(s.shift(-3).rolling(window=2).mean())