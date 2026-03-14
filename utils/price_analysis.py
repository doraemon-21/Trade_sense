import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress

def analyze_price(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")

    hist['returns'] = hist['Close'].pct_change()
    hist = hist.dropna()

    avg_return = hist['returns'].mean() * 252
    volatility = hist['returns'].std() * np.sqrt(252)

    sharpe = avg_return / volatility if volatility != 0 else 0

    cumulative = (1 + hist['returns']).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "current_price": round(hist['Close'].iloc[-1], 2),
        "sharpe_ratio": round(sharpe, 2),
        "volatility": round(volatility * 100, 2),
        "max_drawdown": round(max_drawdown * 100, 2)
    }