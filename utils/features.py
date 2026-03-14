import pandas as pd


# ---------- Technical indicator helpers ----------

def add_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Extend the price history with technical features.

    The following columns are added in-place:
    * MA_5, MA_20 (moving averages)
    * Volatility (10-day rolling std of close)
    * RSI (14-day relative strength index)
    * Return_1, Return_3, Return_5 (lagged pct changes)

    The DataFrame is returned for convenience (same object).
    """

    hist = hist.copy()
    hist["MA_5"] = hist["Close"].rolling(5).mean()
    hist["MA_20"] = hist["Close"].rolling(20).mean()
    hist["Volatility"] = hist["Close"].rolling(10).std()

    # RSI
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    hist["RSI"] = 100 - (100 / (1 + rs))

    # Lagged returns
    hist["Return_1"] = hist["Close"].pct_change()
    hist["Return_3"] = hist["Close"].pct_change(3)
    hist["Return_5"] = hist["Close"].pct_change(5)

    return hist


def merge_sentiment(hist: pd.DataFrame, daily_sentiment: pd.DataFrame) -> pd.DataFrame:
    """Join daily sentiment scores into the history and compute lags.

    ``daily_sentiment`` must contain ``Date`` and ``Sentiment`` columns.
    Missing sentiment days are filled with 0 and the following additional
    features are generated:
    * Sentiment_Lag1 : previous-day score
    * Sentiment_Lag3 : 3-day rolling mean
    * Sentiment_Lag5 : 5-day rolling mean
    """

    if daily_sentiment is None or daily_sentiment.empty:
        hist = hist.copy()
        hist["Sentiment"] = 0
    else:
        hist = hist.merge(daily_sentiment, on="Date", how="left")
        hist["Sentiment"] = hist["Sentiment"].fillna(0)

    hist["Sentiment_Lag1"] = hist["Sentiment"].shift(1)
    hist["Sentiment_Lag3"] = hist["Sentiment"].rolling(3).mean()
    hist["Sentiment_Lag5"] = hist["Sentiment"].rolling(5).mean()

    return hist


def prepare_dataset(hist: pd.DataFrame, daily_sentiment: pd.DataFrame):
    """Produce feature matrix (X) and targets (y_reg, y_clf).

    Returns ``X, y_reg, y_clf, hist`` where ``hist`` is the trimmed history used
    to compute features (rows with NaNs and the last 3 rows removed to align
    targets).
    """

    hist2 = add_technical_indicators(hist)
    hist2 = merge_sentiment(hist2, daily_sentiment)

    # drop rows that still contain NaNs from the feature computation
    hist2 = hist2.dropna()

    features = [
        "MA_5",
        "MA_20",
        "Volatility",
        "RSI",
        "Return_1",
        "Return_3",
        "Return_5",
        "Sentiment",
        "Sentiment_Lag1",
        "Sentiment_Lag3",
        "Sentiment_Lag5",
    ]

    X = hist2[features]
    y_reg = hist2["Close"].shift(-1)
    y_clf = (hist2["Close"].shift(-3) > hist2["Close"]).astype(int)

    # remove the final rows that have been shifted out of the target window
    X = X[:-3]
    y_reg = y_reg[:-3]
    y_clf = y_clf[:-3]
    hist_trimmed = hist2[:-3]

    return X, y_reg, y_clf, hist_trimmed
