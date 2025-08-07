import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# 1. Télécharger les données
ticker = 'BTC-USD'  # Tu peux mettre 'SPY' pour un ETF US
data = yf.download(ticker, start="2021-01-01", end="2025-01-01")
data = data[['Close']]

# 2. Calcul des moyennes mobiles
data['SMA_short'] = data['Close'].rolling(window=20).mean()
data['SMA_long'] = data['Close'].rolling(window=50).mean()

# 3. Génération des signaux
data['Signal'] = 0
data['Signal'][20:] = \
    (data['SMA_short'][20:] > data['SMA_long'][20:]).astype(int)

# 4. Génération des positions (buy = 1, sell = 0)
data['Position'] = data['Signal'].diff()

# 5. Calcul du portefeuille
initial_capital = 10000
data['Returns'] = data['Close'].pct_change()
data['Strategy'] = data['Signal'].shift(1) * data['Returns']
data['Equity'] = (1 + data['Strategy']).cumprod() * initial_capital

# 6. Affichage
plt.figure(figsize=(12, 6))
plt.plot(data['Equity'], label='Stratégie SMA', linewidth=2)
plt.plot((1 + data['Returns']).cumprod() * initial_capital,
         label='Buy & Hold', linestyle='--')
plt.title(f'Stratégie de moyenne mobile sur {ticker}')
plt.legend()
plt.grid()
plt.show()

# 7. Stats
performance = data['Equity'].iloc[-1] / initial_capital - 1
print(f"Performance totale : {performance:.2%}")
