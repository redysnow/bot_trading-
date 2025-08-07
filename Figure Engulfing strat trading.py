import pandas as pd
import yfinance as yf
import ta
import matplotlib.pyplot as plt

# 1. Télécharger les données
symbol = "EURUSD=X"  # Forex EUR/USD
df = yf.download(symbol, start="2022-01-01", interval="1d")
df.dropna(inplace=True)

# 2. Calcul de RSI
df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

# 3. Détection d'engulfing haussier / baissier
def detect_engulfing(df):
    df['engulfing'] = 0
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        
        # Engulfing haussier
        if prev['Close'] < prev['Open'] and curr['Close'] > curr['Open'] and \
           curr['Close'] > prev['Open'] and curr['Open'] < prev['Close']:
            df.loc[df.index[i], 'engulfing'] = 1

        # Engulfing baissier
        elif prev['Close'] > prev['Open'] and curr['Close'] < curr['Open'] and \
             curr['Close'] < prev['Open'] and curr['Open'] > prev['Close']:
            df.loc[df.index[i], 'engulfing'] = -1
    return df

df = detect_engulfing(df)

# 4. Filtre de tendance précédente : 3 bougies dans le sens opposé
def valid_reversal(df, direction):
    result = []
    for i in range(len(df)):
        if i < 4:
            result.append(False)
        else:
            prev = df.iloc[i-4:i-1]
            if direction == 1:  # Long setup
                cond = all(prev['Close'] < prev['Open'])
            else:  # Short setup
                cond = all(prev['Close'] > prev['Open'])
            result.append(cond)
    return result

df['long_setup'] = (df['engulfing'] == 1) & (df['rsi'] < 40) & valid_reversal(df, 1)
df['short_setup'] = (df['engulfing'] == -1) & (df['rsi'] > 60) & valid_reversal(df, -1)

# 5. Simuler les trades
capital = 10000
risk_per_trade = 0.01  # 1%
reward_ratio = 1.5
sl_pct = 0.01  # 1% stop loss
df['position'] = 0
df['returns'] = 0

for i in range(len(df)-1):
    if df.iloc[i]['long_setup']:
        entry = df.iloc[i+1]['Open']
        sl = entry * (1 - sl_pct)
        tp = entry * (1 + sl_pct * reward_ratio)
        for j in range(i+1, len(df)):
            low = df.iloc[j]['Low']
            high = df.iloc[j]['High']
            if low <= sl:
                df.loc[df.index[i], 'returns'] = -risk_per_trade * capital
                break
            elif high >= tp:
                df.loc[df.index[i], 'returns'] = risk_per_trade * capital * reward_ratio
                break
    elif df.iloc[i]['short_setup']:
        entry = df.iloc[i+1]['Open']
        sl = entry * (1 + sl_pct)
        tp = entry * (1 - sl_pct * reward_ratio)
        for j in range(i+1, len(df)):
            high = df.iloc[j]['High']
            low = df.iloc[j]['Low']
            if high >= sl:
                df.loc[df.index[i], 'returns'] = -risk_per_trade * capital
                break
            elif low <= tp:
                df.loc[df.index[i], 'returns'] = risk_per_trade * capital * reward_ratio
                break

# 6. Résultat
df['cumsum'] = df['returns'].cumsum()
print("Total profit:", df['cumsum'].iloc[-1])

# 7. Visualisation
plt.figure(figsize=(12, 6))
plt.plot(df['cumsum'])
plt.title("Cumulative Returns")
plt.grid()
plt.show()
