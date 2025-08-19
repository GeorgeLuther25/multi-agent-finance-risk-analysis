import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from langchain.tools import tool

@tool("get_price_history")
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Returns price history CSV for ticker using yfinance."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return f"ERROR: No data for {ticker}."
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    # Ensure clean column names without ticker prefixes
    df.columns = [col.replace(f"{ticker},", "").strip() if isinstance(col, str) else col for col in df.columns]
    return df.to_csv(index=False)

@tool("get_recent_news")
def get_recent_news(ticker: str, days: int = 14) -> str:
    """Stub news. Replace with your provider later."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    samples = [
        {"date": cutoff, "headline": f"{ticker} announces product update", "sentiment": "positive"},
        {"date": cutoff, "headline": f"{ticker} faces regulatory query", "sentiment": "negative"},
    ]
    return str(samples)