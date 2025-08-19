import ast
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from .config import get_llm
from .tools import get_price_history, get_recent_news
from .schemas import MarketData, NewsBundle, NewsItem, RiskMetrics, RiskReport

from dotenv import load_dotenv
load_dotenv()

# LangSmith visibility
import os
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")

DATA_SYSTEM = "You are the Data Agent. Fetch prices (CSV) and recent news (list). Return raw data only."
RISK_SYSTEM = "You are the Risk Agent. Compute annualized vol, max drawdown, 1D 95% VaR (Gaussian), and a naive Sharpe-like. Add risk flags."
WRITER_SYSTEM = "You are the Writer Agent. Produce a professional Markdown risk report based on inputs."

def _compute_risk(price_csv: str):
    try:
        print(f"DEBUG: price_csv type: {type(price_csv)}")
        print(f"DEBUG: price_csv length: {len(price_csv) if price_csv else 0}")
        print(f"DEBUG: price_csv preview: {price_csv[:200] if price_csv else 'None'}")
        
        if not price_csv or price_csv.strip() == "":
            return {"error": "no_data"}
            
        df = pd.read_csv(StringIO(price_csv))
        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
        
        if "Close" not in df.columns or len(df) < 30:
            return {"error": "insufficient_data"}
            
        df["ret"] = np.log(df["Close"]).diff()
        rets = df["ret"].dropna().values
        mu, sigma = rets.mean(), rets.std(ddof=1)
        ann_vol = float(sigma * np.sqrt(252))
        close = df["Close"].values
        roll_max = np.maximum.accumulate(close)
        drawdown = (close / roll_max) - 1.0
        max_dd = float(drawdown.min())
        var_95 = float(-(mu + sigma * 1.645))
        sharpe_like = float(mu / sigma * np.sqrt(252)) if sigma > 0 else None
        return {"annual_vol": ann_vol, "max_drawdown": max_dd, "daily_var_95": var_95, "sharpe_like": sharpe_like}
    except Exception as e:
        print(f"DEBUG: Error in _compute_risk: {e}")
        return {"error": f"computation_error: {str(e)}"}

class State(BaseModel):
    ticker: str
    period: str = "1y"
    interval: str = "1d"
    horizon_days: int = 30
    market: Optional[MarketData] = None
    news: Optional[NewsBundle] = None
    metrics: Optional[RiskMetrics] = None
    report: Optional[RiskReport] = None

def data_agent(state: State, config: RunnableConfig):
    llm = get_llm()
    _ = llm.invoke([SystemMessage(content=DATA_SYSTEM), HumanMessage(content=f"ticker={state.ticker}")])  # no-op, just for tracing
    price_csv = get_price_history.invoke({"ticker": state.ticker, "period": state.period, "interval": state.interval})
    news_raw = get_recent_news.invoke({"ticker": state.ticker, "days": min(14, state.horizon_days)})
    items = []
    try:
        for r in ast.literal_eval(news_raw):
            items.append(NewsItem(date=str(r["date"]), headline=str(r["headline"]), sentiment=str(r["sentiment"])))
    except Exception:
        pass
    
    # Create new state with updated data
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=MarketData(ticker=state.ticker, period=state.period, interval=state.interval, price_csv=price_csv),
        news=NewsBundle(ticker=state.ticker, window_days=min(14, state.horizon_days), items=items),
        metrics=state.metrics,
        report=state.report
    )
    return new_state

def risk_agent(state: State, config: RunnableConfig):
    llm = get_llm()
    _ = llm.invoke([SystemMessage(content=RISK_SYSTEM), HumanMessage(content="compute risk")])  # tracing
    stats = _compute_risk(state.market.price_csv if state.market else "")
    notes, flags = [], []
    if "error" in stats:
        notes.append(stats["error"])
        stats = {"annual_vol": 0.0, "max_drawdown": 0.0, "daily_var_95": 0.0, "sharpe_like": None}
        flags.append("DATA_QUALITY")
    if stats["annual_vol"] and stats["annual_vol"] > 0.45: flags.append("HIGH_VOLATILITY")
    if stats["max_drawdown"] and stats["max_drawdown"] < -0.25: flags.append("DEEP_DRAWDOWN")

    metrics = RiskMetrics(
        ticker=state.ticker,
        horizon_days=state.horizon_days,
        annual_vol=round(float(stats["annual_vol"]),4),
        max_drawdown=round(float(stats["max_drawdown"]),4),
        daily_var_95=round(float(stats["daily_var_95"]),4),
        sharpe_like=(None if stats.get("sharpe_like") is None else round(float(stats["sharpe_like"]),3)),
        notes=notes,
        risk_flags=sorted(set(flags)),
    )
    
    # Create new state with updated metrics
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=state.market,
        news=state.news,
        metrics=metrics,
        report=state.report
    )
    return new_state

def writer_agent(state: State, config: RunnableConfig):
    llm = get_llm()
    md = f"""# Risk Report — {state.ticker}

**As of:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

## Summary
Automated risk snapshot for {state.ticker} over the past {state.period}. Horizon: {state.horizon_days} days.

## Key Metrics
- Annualized Volatility: **{state.metrics.annual_vol:.4f}**
- Max Drawdown: **{state.metrics.max_drawdown:.4f}**
- 1D VaR (95%): **{state.metrics.daily_var_95:.4f}**
- Sharpe-like: **{state.metrics.sharpe_like}**

## Risk Flags
{', '.join(state.metrics.risk_flags) if state.metrics.risk_flags else 'None'}

## Recent News (stub)
{chr(10).join(f"- {n.date}: {n.headline} [{n.sentiment}]" for n in (state.news.items if state.news else []))}

## Methodology
- Prices from yfinance; log returns
- Annualized vol = std(returns)*sqrt(252)
- Max drawdown = min(Price/Peak - 1)
- VaR(95%) = -(μ + 1.645σ), Gaussian
"""
    _ = llm.invoke([SystemMessage(content=WRITER_SYSTEM), HumanMessage(content="draft report")])  # tracing
    report = RiskReport(
        ticker=state.ticker,
        as_of=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        summary=f"Risk snapshot for {state.ticker}.",
        key_findings=["Automated metrics computed from historical data."],
        metrics_table={
            "annual_vol": state.metrics.annual_vol,
            "max_drawdown": state.metrics.max_drawdown,
            "daily_var_95": state.metrics.daily_var_95,
            "sharpe_like": state.metrics.sharpe_like,
        },
        risk_flags=state.metrics.risk_flags,
        methodology="Gaussian VaR; log returns; daily OHLC from yfinance.",
        markdown_report=md,
    )
    
    # Create new state with updated report
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=state.market,
        news=state.news,
        metrics=state.metrics,
        report=report
    )
    return new_state