import yfinance as yf
import pandas as pd


def fetch_price_history(ticker: str, start: str = None, end: str = None, period: str = "1y") -> pd.DataFrame:
    """Return a DataFrame of historical price data for *ticker*.

    - If *start* or *end* are specified, those dates are used.
    - Otherwise the ``period`` argument is passed to ``yfinance.download``.

    The returned frame will have a normalized ``Date`` column (tz naive) and
    flattened column names (no MultiIndex).
    If no data is available an empty DataFrame is returned.
    """

    if start or end:
        hist = yf.download(ticker, start=start, end=end, progress=False)
    else:
        hist = yf.download(ticker, period=period, progress=False)

    if hist is None or hist.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns when multiple tickers are
    # requested or on certain download configurations. Flatten them.
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist = hist.reset_index()
    hist["Date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
    return hist
