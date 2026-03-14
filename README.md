# TradeSense — AI-Powered Stock Analysis Platform

TradeSense is a Flask web application that combines machine learning and NLP to analyze stocks, predict price movements, and surface news sentiment — all in one dashboard.

---

## Features

- **Price Prediction** — XGBoost regression model predicts next-day closing price
- **Trade Signal** — XGBoost classifier outputs a BUY / HOLD / SELL recommendation with confidence probability
- **News Sentiment** — FinBERT (ProsusAI) scores financial news headlines fetched from the Finnhub API
- **Technical Indicators** — MA5, MA20, RSI, Volatility, and lagged returns used as model features
- **Interactive Charts** — Plotly price history charts with a dark theme
- **Sentiment Dashboard** — Dedicated page showing positive/negative news breakdown and sentiment trend over time

---

## Supported Stocks

| Ticker | Company |
|--------|---------|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corp. |
| GOOGL | Alphabet Inc. |
| AMZN | Amazon.com Inc. |
| TSLA | Tesla Inc. |
| META | Meta Platforms Inc. |
| NVDA | Nvidia Corp. |
| NFLX | Netflix Inc. |

---

## Project Structure

```
capstone/
├── app.py                  # Flask application & routes
├── config.py               # API key configuration (not committed)
├── requirements.txt        # Python dependencies
├── app/
│   ├── static/css/
│   │   └── style.css       # Custom dark-theme styles
│   └── templates/
│       ├── index.html      # Stock listing page
│       ├── dashboard.html  # Prediction & analysis dashboard
│       └── sentiment.html  # News sentiment page
└── utils/
    ├── data_loader.py      # yfinance price history fetcher
    ├── features.py         # Technical indicators & sentiment merging
    ├── modeling.py         # XGBoost train/evaluate helpers
    ├── news_analysis.py    # Finnhub API + FinBERT sentiment pipeline
    └── price_analysis.py   # Sharpe ratio, volatility, drawdown metrics
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/doraemon-21/Trade_sense.git
cd Trade_sense
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `config.py` file in the root directory:

```python
FINNHUB_API_KEY = "your_finnhub_api_key_here"
```

Get a free API key at [finnhub.io](https://finnhub.io).

### 5. Run the app

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## Analysis

analyze.png

## How It Works

1. **Select a stock** from the home page
2. Click **Analyze** — the app fetches 1 year of price data via `yfinance`
3. News articles are pulled from Finnhub and scored using **FinBERT** (a finance-specific BERT model)
4. Technical indicators and sentiment scores are combined into a feature matrix
5. Two XGBoost models are trained on the fly:
   - **Regressor** → predicted next closing price
   - **Classifier** → probability the price goes up (→ BUY/HOLD/SELL signal)
6. Results, charts, and top feature importances are displayed on the dashboard

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | Flask |
| ML Models | XGBoost |
| NLP / Sentiment | HuggingFace Transformers (FinBERT) |
| Price Data | yfinance |
| News Data | Finnhub API |
| Charts | Plotly |
| Data Processing | pandas, numpy, scikit-learn, scipy |

---

## Notes

- `config.py` is excluded from version control — never commit your API key
- The FinBERT model is downloaded from HuggingFace on first run (~500MB)
- Model training happens on every request; no pre-trained models are saved to disk
