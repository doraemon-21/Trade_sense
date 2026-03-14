import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
from config import FINNHUB_API_KEY

# ---------------------------------------------------
# Load FinBERT Once (Avoid Reloading Every Time)
# ---------------------------------------------------
try:
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert"
    )
except Exception:
    sentiment_model = None


# ---------------------------------------------------
# Production-Safe News + Sentiment Function
# ---------------------------------------------------
def get_daily_sentiment(ticker, start: datetime = None, end: datetime = None, lookback_days: int = 60):
    """Fetch and score news for *ticker*.

    ``start``/``end`` may be provided as ``datetime`` objects; if omitted the
    function will look back ``lookback_days`` from today.  The behaviour is
    identical to the original version when called with just the ticker.
    """

    # Default safe returns
    empty_df = pd.DataFrame(columns=["Date", "Sentiment"])
    positive_news = []
    negative_news = []
    articles_info = []  # will hold detailed records for sentiment page

    # Validate API Key
    if not FINNHUB_API_KEY or FINNHUB_API_KEY.strip() == "" or FINNHUB_API_KEY == "YOUR_API_KEY_HERE":
        # avoid fancy unicode in case the terminal encoding is limited
        print("WARNING: FINNHUB_API_KEY is missing or unset.  News sentiment will be disabled.")
        return empty_df, positive_news, negative_news, pd.DataFrame(articles_info)

    today = datetime.today()
    if start is None or end is None:
        # fall back to lookback window if explicit dates aren't supplied
        if end is None:
            end = today
        if start is None:
            start = end - timedelta(days=lookback_days)

    url = (
        f"https://finnhub.io/api/v1/company-news?"
        f"symbol={ticker}"
        f"&from={start.date()}"
        f"&to={end.date()}"
        f"&token={FINNHUB_API_KEY}"
    )

    # ---------------------------------------------------
    # Safe API Call
    # ---------------------------------------------------
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            # log the failure without Unicode symbols
            print(f"WARNING: Finnhub API returned status {response.status_code}. Check your API key and network.")
            return empty_df, positive_news, negative_news, pd.DataFrame(articles_info)

        try:
            articles = response.json()
        except Exception:
            print("⚠ Failed to decode JSON from Finnhub.")
            return empty_df, positive_news, negative_news, pd.DataFrame(articles_info)

    except requests.exceptions.RequestException as e:
        print(f"⚠ Network Error: {e}")
        # keep return signature consistent with callers
        return empty_df, positive_news, negative_news, pd.DataFrame(articles_info)

    # Validate response structure
    if not isinstance(articles, list) or len(articles) == 0:
        print("⚠ No articles returned from Finnhub.")
        return empty_df, positive_news, negative_news, pd.DataFrame(articles_info)

    # ---------------------------------------------------
    # Process Articles
    # ---------------------------------------------------
    data = []

    for article in articles:

        headline = article.get("headline", "")
        summary = article.get("summary", "")
        source = article.get("source", "Unknown")
        timestamp = article.get("datetime", None)

        if not timestamp:
            continue

        date = datetime.fromtimestamp(timestamp).date()
        text = (headline + " " + summary).strip()

        if len(text) < 20:
            continue

        # If sentiment model failed to load
        if sentiment_model is None:
            continue

        try:
            result = sentiment_model(text[:512])[0]
        except Exception:
            continue

        label = result.get("label", "").lower()

        if label == "positive":
            score = 1
            positive_news.append((headline, source))
        elif label == "negative":
            score = -1
            negative_news.append((headline, source))
        else:
            score = 0

        data.append({
            "Date": pd.to_datetime(date),
            "Score": score
        })

        # also keep a full record for downstream charts/listing
        articles_info.append({
            "Date": pd.to_datetime(date),
            "Score": score,
            "label": label,
            "score_val": result.get("score", 0),
            "headline": headline,
            "source": source,
        })

    if len(data) == 0:
        # no scored articles, return empty articles dataframe as well
        return empty_df, positive_news, negative_news, pd.DataFrame(articles_info)

    df = pd.DataFrame(data)
    articles_df = pd.DataFrame(articles_info)

    # ---------------------------------------------------
    # Aggregate Daily Sentiment
    # ---------------------------------------------------
    daily_sentiment = (
        df.groupby("Date")["Score"]
        .mean()
        .reset_index()
    )

    daily_sentiment.columns = ["Date", "Sentiment"]

    return daily_sentiment, positive_news, negative_news, articles_df