import ast
import json
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from config import get_llm
from tools import get_price_history, get_recent_news, query_10k_documents
from schemas import (
    MarketData, NewsBundle, NewsItem, RiskMetrics, RiskReport,
    SentimentSummary, ValuationMetrics, FundamentalAnalysis
)
from rag_utils import initialize_sample_data, FundamentalRAG, batch_ingest_documents

from dotenv import load_dotenv
load_dotenv()

# LangSmith visibility
import os
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")

DATA_SYSTEM = "You are the Data Agent. Fetch prices (CSV) and recent news (list). Return raw data only."
RISK_SYSTEM = "You are the Risk Agent. Compute annualized vol, max drawdown, 1D 95% VaR (Gaussian), and a naive Sharpe-like. Add risk flags."
SENTIMENT_SYSTEM = """As a sentiment equity analyst your primary responsibility is to analyze the financial news, analyst ratings and disclosures related to the underlying security, and analyze its implication and sentiment for investors or stakeholders. You analyze financial news using reflection-enhanced prompting.

Your task is to:
1. SUMMARIZE: First, provide a concise summary of each news item
2. CRITIQUE: Evaluate the quality and relevance of your summary
3. REFINE: Improve your analysis based on the critique
4. CONCLUDE: Provide an overall sentiment analysis and investment recommendation

For each news item, consider:
- Market impact and relevance
- Sentiment indicators (positive, negative, neutral language)
- Financial implications
- Credibility of the source

Your final output should include:
- Overall sentiment (bullish/bearish/neutral)
- Confidence score (0.0-1.0)
- Investment recommendation with reasoning
- Key insights from the news analysis

Use reflection to ensure your analysis is thorough and well-reasoned."""
VALUATION_SYSTEM = """As a valuation equity analyst, your primary responsibility is to analyze the valuation trends of a given asset or portfolio over an extended time horizon. To complete the task, you must analyze the historical valuation data of the asset or portfolio provided, identify trends and patterns in valuation metrics over time, and interpret the implications of these trends for investors or stakeholders.

Focus your analysis on:
1. Price trend analysis (upward, downward, sideways movement)
2. Volatility regime assessment (low, medium, high volatility periods)
3. Risk-return profile evaluation
4. Investment implications and outlook
5. Key patterns and inflection points in the data

Provide clear, actionable insights based on the computational metrics provided."""

FUNDAMENTAL_SYSTEM = """As a fundamental financial equity analyst your primary
responsibility is to analyze the most recent 10K report provided for a company.
You have access to a powerful tool that can help you extract relevant information
from the 10K. Your analysis should be based solely on the information that you
retrieve using this tool. You can interact with this tool using natural language
queries. The tool will understand your requests and return relevant text snippets
and data points from the 10K document. Keep checking if you have answered the
users' question to avoid looping."""

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
    sentiment: Optional[SentimentSummary] = None
    valuation: Optional[ValuationMetrics] = None
    fundamental: Optional[FundamentalAnalysis] = None
    metrics: Optional[RiskMetrics] = None
    report: Optional[RiskReport] = None

def data_agent(state: State, config: RunnableConfig):
    # llm = get_llm()
    # _ = llm.invoke([SystemMessage(content=DATA_SYSTEM), HumanMessage(content=f"ticker={state.ticker}")])  # no-op, just for tracing
    price_csv = get_price_history.invoke({"ticker": state.ticker, "period": state.period, "interval": state.interval})
    news_raw = get_recent_news.invoke({"ticker": state.ticker, "days": min(14, state.horizon_days)})
    items = []
    try:
        for r in ast.literal_eval(news_raw):
            # print("News is:", r["content"])
            items.append(NewsItem(date=str(r["date"]), headline=str(r["headline"]), sentiment=str(r["sentiment"]), content=str(r["content"])))
    except Exception as e:
        print(f"Exception occured:{e}")
    
    # Create new state with updated data
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=MarketData(
            ticker=state.ticker,
            period=state.period,
            interval=state.interval,
            price_csv=price_csv
        ),
        news=NewsBundle(
            ticker=state.ticker,
            window_days=min(14, state.horizon_days),
            items=items
        ),
        sentiment=state.sentiment,
        valuation=state.valuation,
        fundamental=state.fundamental,
        metrics=state.metrics,
        report=state.report
    )
    return new_state


def sentiment_agent(state: State, config: RunnableConfig):
    """
    Sentiment agent that analyzes news using reflection-enhanced prompting.
    Implements a multi-step process: summarize, critique, refine, conclude.
    """
    llm = get_llm()
    
    if not state.news or not state.news.items:
        # No news data available
        sentiment_summary = SentimentSummary(
            ticker=state.ticker,
            news_items_analyzed=0,
            overall_sentiment="neutral",
            confidence_score=0.0,
            summary="No news data available for analysis.",
            investment_recommendation="Cannot provide recommendation due to lack of news data.",
            key_insights=["No news items found for analysis"],
            methodology="LLM-based reflection-enhanced summarization"
        )
        
        new_state = State(
            ticker=state.ticker,
            period=state.period,
            interval=state.interval,
            horizon_days=state.horizon_days,
            market=state.market,
            news=state.news,
            sentiment=sentiment_summary,
            valuation=state.valuation,
            metrics=state.metrics,
            report=state.report
        )
        return new_state
    
    # Prepare news data for analysis
    news_text = []
    for item in state.news.items:
        news_text.append(f"Date: {item.date}\nHeadline: {item.headline}\nContent: {item.content}")
    
    news_content = "\n\n".join(news_text)
    
    # Step 1: Initial Analysis with Reflection-Enhanced Prompting
    analysis_prompt = f"""
    Analyze the following news items for {state.ticker} using reflection-enhanced prompting:

    NEWS DATA:
    {news_content}

    PROCESS:
    1. SUMMARIZE: First, provide a concise summary of each news item and its potential market impact.
    2. CRITIQUE: Evaluate your summary - is it comprehensive? Are you missing key insights? What biases might be present?
    3. REFINE: Based on your critique, improve and refine your analysis.
    4. CONCLUDE: Provide your final assessment.

    Your response should also include:
    - Overall sentiment analysis (bullish/bearish/neutral)
    - Confidence level (0.0 to 1.0)
    - Key insights and reasoning
    - Investment recommendation with clear rationale

    Provide a concise summary along with an informed recommendation on whether to invest in this stock.
    """
    
    try:
        response = llm.invoke([
            SystemMessage(content=SENTIMENT_SYSTEM),
            HumanMessage(content=analysis_prompt)
        ])
        response_text = response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"⚠️  Sentiment LLM call failed: {e}")
        # Fallback: build a simple analysis from the raw news items
        summaries = []
        for item in state.news.items if state.news and state.news.items else []:
            sentiment = item.sentiment.lower()
            summaries.append(f"{item.date}: {item.headline} (sentiment: {sentiment})")
        response_text = (
            "Sentiment analysis fallback due to LLM error.\n"
            + "\n".join(summaries) if summaries else
            "Sentiment analysis fallback triggered but no news items were available."
        )
    
    # Simple sentiment extraction (could be enhanced with more sophisticated parsing)
    sentiment_mapping = {
        "bullish": "bullish",
        "bearish": "bearish", 
        "neutral": "neutral",
        "positive": "bullish",
        "negative": "bearish"
    }
    
    overall_sentiment = "neutral"
    confidence_score = 0.5
    
    # Extract sentiment from response
    response_lower = response_text.lower()
    for keyword, sentiment in sentiment_mapping.items():
        if keyword in response_lower:
            overall_sentiment = sentiment
            break
    
    # Extract confidence (simple heuristic)
    if "high confidence" in response_lower or "very confident" in response_lower:
        confidence_score = 0.8
    elif "low confidence" in response_lower or "uncertain" in response_lower:
        confidence_score = 0.3
    elif "moderate" in response_lower or "medium" in response_lower:
        confidence_score = 0.6
    
    # Extract key insights (simple extraction based on common patterns)
    key_insights = []
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('-') or line.startswith('•') or 'insight' in line.lower():
            key_insights.append(line.strip('- •').strip())
    
    if not key_insights:
        key_insights = ["Analysis completed using reflection-enhanced prompting methodology"]
    
    sentiment_summary = SentimentSummary(
        ticker=state.ticker,
        news_items_analyzed=len(state.news.items),
        overall_sentiment=overall_sentiment,
        confidence_score=confidence_score,
        # summary=response_text[:500] + "..." if len(response_text) > 500 else response_text,
        summary=response_text,
        investment_recommendation=f"Based on sentiment analysis: {overall_sentiment} outlook with {confidence_score:.1%} confidence",
        key_insights=key_insights[:5],  # Limit to top 5 insights
        methodology="LLM-based reflection-enhanced summarization"
    )
    
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=state.market,
        news=state.news,
        sentiment=sentiment_summary,
        valuation=state.valuation,
        fundamental=state.fundamental,
        metrics=state.metrics,
        report=state.report
    )
    return new_state


def fundamental_agent(state: State, config: RunnableConfig):
    """
    Fundamental agent that analyzes 10-K/10-Q data using RAG as a tool.
    """
    # Initialize RAG system to ensure sample data exists
    rag_system = FundamentalRAG()
    available_filings = rag_system.get_available_filings(state.ticker)
    if not available_filings:
        print(f"No filings found for {state.ticker}, initializing sample data...")
        # initialize_sample_data(rag_system)
        batch_ingest_documents(rag_system, directory="./data/filings")
        available_filings = rag_system.get_available_filings(state.ticker)
    
    if not available_filings:
        # Create a basic fundamental analysis with no data
        fundamental_analysis = FundamentalAnalysis(
            ticker=state.ticker,
            filing_type="N/A",
            filing_date="N/A",
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            executive_summary=f"No 10-K/10-Q data available for {state.ticker}",
            key_financial_metrics={},
            business_highlights=[],
            risk_factors=["No data available"],
            competitive_position="Unable to assess due to lack of data",
            growth_prospects="Unable to assess due to lack of data",
            financial_health_score=5.0,
            investment_thesis="Cannot provide investment thesis without fundamental data",
            concerns_and_risks=["No fundamental data available for analysis"],
            methodology="RAG-enhanced 10-K/10-Q document analysis"
        )
    else:
        llm = get_llm()
        default_queries = [
            "executive summary",
            "financial metrics",
            "business segments",
            "risk factors",
            "competitive position",
            "growth prospects",
            "investment outlook"
        ]
        query_payload = ", ".join(default_queries)
        try:
            rag_result = query_10k_documents.invoke({
                "ticker": state.ticker,
                "query": query_payload
            })
        except Exception as e:
            print(f"query_10k_documents invocation failed: {e}")
            rag_result = ""

        if isinstance(rag_result, list):
            context = "\n\n".join(str(chunk) for chunk in rag_result)
        else:
            context = str(rag_result or "")

        if not context.strip() or context.strip().upper().startswith("ERROR"):
            context = "No substantive filing excerpts were retrieved."

        analysis_messages = [
            SystemMessage(content=f"""
{FUNDAMENTAL_SYSTEM}

Analyze the provided filing excerpts for {state.ticker}. Return a compact JSON object using this schema:
{{
  "executive_summary": "...",
  "key_financial_metrics": {{"metric": "value", "...": "..."}},
  "business_highlights": ["..."],
  "risk_factors": ["..."],
  "competitive_position": "...",
  "growth_prospects": "...",
  "financial_health_score": 0.0,
  "investment_thesis": "...",
  "concerns_and_risks": ["..."]
}}
Ensure the response is valid JSON and keep numeric scores between 0 and 10.
"""),
            HumanMessage(content=f"Filing excerpts for {state.ticker}:\n{context}\n\nProduce the JSON now.")
        ]

        try:
            model_response = llm.invoke(analysis_messages, config=config)
            analysis_content = model_response.content if hasattr(model_response, "content") else str(model_response)
        except Exception as e:
            print(f"LLM fundamental analysis error: {e}")
            analysis_content = ""

        parsed = {}
        if analysis_content:
            try:
                parsed = json.loads(analysis_content)
            except json.JSONDecodeError:
                print("Fundamental analysis did not return JSON; using fallback text.")

        filing_info = available_filings[0]

        executive_summary = parsed.get("executive_summary") if isinstance(parsed, dict) else None
        key_metrics = parsed.get("key_financial_metrics") if isinstance(parsed, dict) else None
        business_highlights = parsed.get("business_highlights") if isinstance(parsed, dict) else None
        risk_factors = parsed.get("risk_factors") if isinstance(parsed, dict) else None
        competitive_position = parsed.get("competitive_position") if isinstance(parsed, dict) else None
        growth_prospects = parsed.get("growth_prospects") if isinstance(parsed, dict) else None
        investment_thesis = parsed.get("investment_thesis") if isinstance(parsed, dict) else None
        concerns_and_risks = parsed.get("concerns_and_risks") if isinstance(parsed, dict) else None

        try:
            score = float(parsed.get("financial_health_score", 6.5)) if isinstance(parsed, dict) else 6.5
        except (TypeError, ValueError):
            score = 6.5
        score = min(max(score, 0.0), 10.0)

        filing_info = available_filings[0]
        
        # Create structured fundamental analysis
        fundamental_analysis = FundamentalAnalysis(
            ticker=state.ticker,
            filing_type=filing_info.get("filing_type", "10-K"),
            filing_date=filing_info.get("ingestion_date", "Unknown"),
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            executive_summary=executive_summary or analysis_content or "Fundamental analysis generated without structured details.",
            key_financial_metrics=key_metrics if isinstance(key_metrics, dict) else {"summary": analysis_content or "No metrics extracted."},
            business_highlights=business_highlights if isinstance(business_highlights, list) else [
                "Automated summary from SEC filings"
            ],
            risk_factors=risk_factors if isinstance(risk_factors, list) else [
                "Risk factors could not be parsed from filings."
            ],
            competitive_position=competitive_position or "Competitive positioning insights were inferred from filings.",
            growth_prospects=growth_prospects or "Growth outlook inferred from available filing excerpts.",
            financial_health_score=score,
            investment_thesis=investment_thesis or f"Analysis synthesized from {state.ticker}'s SEC filings.",
            concerns_and_risks=concerns_and_risks if isinstance(concerns_and_risks, list) else [
                "Further review of filings recommended."
            ],
            methodology="RAG-enhanced 10-K/10-Q document analysis"
        )
    
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=state.market,
        news=state.news,
        sentiment=state.sentiment,
        valuation=state.valuation,
        fundamental=fundamental_analysis,
        metrics=state.metrics,
        report=state.report
    )
    return new_state


def _compute_valuation_metrics(price_csv: str, ticker: str, period: str) -> ValuationMetrics:
    """
    Compute valuation metrics including annualized return and volatility.
    Uses the formulas specified:
    - R_annualized = ((1 + R_cumulative)^(252/n)) - 1
    - σ_annualized = σ_daily × √252
    """
    try:
        # Parse the CSV data
        df = pd.read_csv(StringIO(price_csv))
        
        # Ensure we have required columns
        if 'Close' not in df.columns or 'Date' not in df.columns:
            raise ValueError("CSV must contain 'Close' and 'Date' columns")
        
        # Sort by date to ensure proper order
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Calculate daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Remove any NaN values
        daily_returns = df['Daily_Return'].dropna()
        
        if len(daily_returns) < 2:
            raise ValueError("Insufficient data for valuation analysis")
        
        # Number of trading days
        n = len(daily_returns)
        
        # Calculate cumulative return
        start_price = df['Close'].iloc[0]
        end_price = df['Close'].iloc[-1]
        cumulative_return = (end_price - start_price) / start_price
        
        # Calculate annualized return: R_annualized = ((1 + R_cumulative)^(252/n)) - 1
        annualized_return = ((1 + cumulative_return) ** (252 / n)) - 1
        
        # Calculate daily volatility (standard deviation of daily returns)
        daily_volatility = daily_returns.std()
        
        # Calculate annualized volatility: σ_annualized = σ_daily × √252
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Determine price trend
        price_change_pct = (end_price - start_price) / start_price
        if price_change_pct > 0.05:  # 5% threshold
            price_trend = "upward"
        elif price_change_pct < -0.05:
            price_trend = "downward"
        else:
            price_trend = "sideways"
        
        # Determine volatility regime
        if annualized_volatility < 0.15:  # 15% annual volatility
            volatility_regime = "low"
        elif annualized_volatility < 0.30:  # 30% annual volatility
            volatility_regime = "medium"
        else:
            volatility_regime = "high"
        
        # Generate insights
        insights = []
        insights.append(f"Price moved {price_change_pct:.2%} over {n} trading days")
        insights.append(f"Daily volatility of {daily_volatility:.4f} ({annualized_volatility:.2%} annualized)")
        
        if annualized_return > 0.10:
            insights.append("Strong positive annualized returns")
        elif annualized_return < -0.10:
            insights.append("Negative annualized returns indicate underperformance")
        
        if volatility_regime == "high":
            insights.append("High volatility suggests increased investment risk")
        elif volatility_regime == "low":
            insights.append("Low volatility indicates stable price movement")
        
        # Risk-return assessment
        if annualized_return > 0 and volatility_regime == "low":
            risk_assessment = "Favorable risk-return profile with positive returns and low volatility"
        elif annualized_return > 0 and volatility_regime == "high":
            risk_assessment = "High-risk, high-reward profile with positive returns but elevated volatility"
        elif annualized_return < 0 and volatility_regime == "high":
            risk_assessment = "Unfavorable profile with negative returns and high volatility"
        else:
            risk_assessment = "Mixed risk profile requiring careful evaluation"
        
        # Trend analysis
        trend_analysis = f"The {ticker} exhibits a {price_trend} trend over the analysis period with {volatility_regime} volatility regime."
        
        return ValuationMetrics(
            ticker=ticker,
            analysis_period=period,
            trading_days=n,
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            daily_volatility=daily_volatility,
            annualized_volatility=annualized_volatility,
            price_trend=price_trend,
            volatility_regime=volatility_regime,
            valuation_insights=insights,
            trend_analysis=trend_analysis,
            risk_assessment=risk_assessment
        )
        
    except Exception as e:
        print(f"Error computing valuation metrics: {e}")
        # Return default metrics on error
        return ValuationMetrics(
            ticker=ticker,
            analysis_period=period,
            trading_days=0,
            cumulative_return=0.0,
            annualized_return=0.0,
            daily_volatility=0.0,
            annualized_volatility=0.0,
            price_trend="sideways",
            volatility_regime="medium",
            valuation_insights=["Error in calculation - insufficient data"],
            trend_analysis="Unable to determine trend due to data issues",
            risk_assessment="Cannot assess risk due to insufficient data"
        )


def valuation_agent(state: State, config: RunnableConfig):
    """
    Valuation agent that analyzes historical price data to compute valuation metrics
    and trends using computational tools for volatility and return calculations.
    """
    llm = get_llm()
    
    # Check if we have market data
    if not state.market or not state.market.price_csv:
        print("⚠️  No market data available for valuation analysis")
        # Create default valuation metrics
        valuation_metrics = ValuationMetrics(
            ticker=state.ticker,
            analysis_period=state.period,
            trading_days=0,
            cumulative_return=0.0,
            annualized_return=0.0,
            daily_volatility=0.0,
            annualized_volatility=0.0,
            price_trend="sideways",
            volatility_regime="medium",
            valuation_insights=["No market data available for analysis"],
            trend_analysis="Cannot analyze trends without price data",
            risk_assessment="Unable to assess risk without market data"
        )
    else:
        # Compute valuation metrics using computational tools
        print(f"📊 Computing valuation metrics for {state.ticker}...")
        valuation_metrics = _compute_valuation_metrics(
            state.market.price_csv, 
            state.ticker, 
            state.period
        )
        
        # Use LLM for enhanced trend analysis and interpretation
        analysis_prompt = f"""
        Based on the computed valuation metrics for {state.ticker}:
        
        - Analysis Period: {valuation_metrics.analysis_period}
        - Trading Days: {valuation_metrics.trading_days}
        - Cumulative Return: {valuation_metrics.cumulative_return:.4f} ({valuation_metrics.cumulative_return:.2%})
        - Annualized Return: {valuation_metrics.annualized_return:.4f} ({valuation_metrics.annualized_return:.2%})
        - Daily Volatility: {valuation_metrics.daily_volatility:.6f}
        - Annualized Volatility: {valuation_metrics.annualized_volatility:.4f} ({valuation_metrics.annualized_volatility:.2%})
        - Price Trend: {valuation_metrics.price_trend}
        - Volatility Regime: {valuation_metrics.volatility_regime}
        - Market Price: {state.market.price_csv}
        
        Provide enhanced trend analysis and investment implications based on these metrics.
        """
        
        try:
            response = llm.invoke([
                SystemMessage(content=VALUATION_SYSTEM),
                HumanMessage(content=analysis_prompt)
            ])

            enhanced_analysis = response.content if hasattr(response, 'content') else str(response)
            valuation_metrics.trend_analysis = enhanced_analysis

        except Exception as e:
            print(f"⚠️  Error in LLM analysis: {e}")
            valuation_metrics.trend_analysis = (
                valuation_metrics.trend_analysis
                + "\n\nLLM commentary unavailable; using computed metrics only."
            )
    
    # Create new state with valuation metrics
    new_state = State(
        ticker=state.ticker,
        period=state.period,
        interval=state.interval,
        horizon_days=state.horizon_days,
        market=state.market,
        news=state.news,
        sentiment=state.sentiment,
        valuation=valuation_metrics,
        metrics=state.metrics,
        report=state.report
    )
    return new_state


def risk_agent(state: State, config: RunnableConfig):
    llm = get_llm()
    try:
        _ = llm.invoke([SystemMessage(content=RISK_SYSTEM), HumanMessage(content="compute risk")])  # tracing
    except Exception as e:
        print(f"⚠️  Risk LLM call skipped: {e}")
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
        sentiment=state.sentiment,
        valuation=state.valuation,
        metrics=metrics,
        report=state.report
    )
    return new_state

def writer_agent(state: State, config: RunnableConfig):
    # llm = get_llm()
    
    # Build sentiment section
    sentiment_section = ""
    if state.sentiment:
        sentiment_section = f"""
## Sentiment Analysis
- **Overall Sentiment**: {state.sentiment.overall_sentiment.title()}
- **Confidence Score**: {state.sentiment.confidence_score:.1%}
- **News Items Analyzed**: {state.sentiment.news_items_analyzed}
- **Investment Recommendation**: {state.sentiment.investment_recommendation}

### Key Insights
{chr(10).join(f"- {insight}" for insight in state.sentiment.key_insights)}

### Summary
{state.sentiment.summary}
"""
    else:
        sentiment_section = """
## Sentiment Analysis
No sentiment analysis available - insufficient news data.
"""

    # Build valuation section
    valuation_section = ""
    if state.valuation:
        valuation_section = f"""
## Valuation Analysis
- **Analysis Period**: {state.valuation.analysis_period}
- **Trading Days Analyzed**: {state.valuation.trading_days}
- **Cumulative Return**: {state.valuation.cumulative_return:.4f} ({state.valuation.cumulative_return:.2%})
- **Annualized Return**: {state.valuation.annualized_return:.4f} ({state.valuation.annualized_return:.2%})
- **Daily Volatility**: {state.valuation.daily_volatility:.6f}
- **Annualized Volatility**: {state.valuation.annualized_volatility:.4f} ({state.valuation.annualized_volatility:.2%})
- **Price Trend**: {state.valuation.price_trend.title()}
- **Volatility Regime**: {state.valuation.volatility_regime.title()}

### Valuation Insights
{chr(10).join(f"- {insight}" for insight in state.valuation.valuation_insights)}

### Trend Analysis
{state.valuation.trend_analysis}

### Risk Assessment
{state.valuation.risk_assessment}
"""
    else:
        valuation_section = """
## Valuation Analysis
No valuation analysis available - insufficient market data.
"""

    # Build fundamental analysis section
    fundamental_section = ""
    if state.fundamental:
        fundamental_section = f"""
## Fundamental Analysis (10-K/10-Q Based)
- **Filing Type**: {state.fundamental.filing_type}
- **Filing Date**: {state.fundamental.filing_date}
- **Financial Health Score**: {state.fundamental.financial_health_score:.1f}/10.0

### Executive Summary
{state.fundamental.executive_summary}

### Business Highlights
{chr(10).join(f"- {highlight}" for highlight in state.fundamental.business_highlights)}

### Risk Factors
{chr(10).join(f"- {risk}" for risk in state.fundamental.risk_factors)}

### Competitive Position
{state.fundamental.competitive_position}

### Growth Prospects
{state.fundamental.growth_prospects}

### Investment Thesis
{state.fundamental.investment_thesis}

### Concerns and Risks
{chr(10).join(f"- {concern}" for concern in state.fundamental.concerns_and_risks)}
"""
    else:
        fundamental_section = """
## Fundamental Analysis
No fundamental analysis available - no 10-K/10-Q data found.
"""

    current_ts = datetime.utcnow().strftime('%b %d, %Y at %I:%M %p UTC').replace(' 0', ' ').lstrip('0')

    metrics_obj = state.metrics
    valuation_obj = state.valuation
    sentiment_obj = state.sentiment
    fundamental_obj = state.fundamental

    if metrics_obj and metrics_obj.risk_flags:
        risk_summary = "Risk watchlist flagged: " + ", ".join(metrics_obj.risk_flags)
    else:
        risk_summary = "Key risk indicators are within typical ranges."

    volatility_label = "unknown"
    if metrics_obj and metrics_obj.annual_vol is not None:
        volatility_label = "low" if metrics_obj.annual_vol < 0.15 else "elevated"

    if valuation_obj:
        valuation_trend = valuation_obj.price_trend
        valuation_tone = (
            "Trend shows constructive momentum."
            if valuation_obj.price_trend != "downward"
            else "Trend pressure leans negative; review positioning."
        )
    else:
        valuation_trend = "Price momentum unclear"
        valuation_tone = "Insufficient data to characterise trend."

    if sentiment_obj:
        if sentiment_obj.overall_sentiment == "bullish":
            sentiment_tone = "Sentiment skews bullish."
        elif sentiment_obj.overall_sentiment == "bearish":
            sentiment_tone = "Sentiment caution persists."
        else:
            sentiment_tone = "Sentiment mix appears balanced."
    else:
        sentiment_tone = "Sentiment data limited."

    if fundamental_obj:
        if fundamental_obj.financial_health_score >= 7:
            fundamental_tone = (
                f"Financial health score {fundamental_obj.financial_health_score:.1f}/10 reflects solid fundamentals."
            )
        else:
            fundamental_tone = (
                f"Financial health score {fundamental_obj.financial_health_score:.1f}/10 highlights areas to monitor."
            )
    else:
        fundamental_tone = "Fundamental details unavailable."

    if sentiment_obj and sentiment_obj.overall_sentiment == "bullish":
        bottom_line = "Position looks resilient but stay selective."
    else:
        bottom_line = "Maintain watchful posture and reassess catalysts."

    news_entries = []
    if state.news and state.news.items:
        for item in state.news.items:
            news_entries.append(f"- {item.date}: {item.headline} [{item.sentiment}]")
    news_section = "\n".join(news_entries) if news_entries else "No recent headlines captured."

    notes_text = (
        ", ".join(metrics_obj.notes) if metrics_obj and metrics_obj.notes else "No unusual observations logged."
    )

    md = f"""# {state.ticker} Investment Brief
**Last refreshed:** {current_ts}

---

## Snapshot
| Lens | Takeaway |
| --- | --- |
| Price action | {valuation_tone} |
| Risk posture | {risk_summary} |
| Sentiment pulse | {sentiment_tone} |
| Fundamentals | {fundamental_tone} |

---

## Executive Dashboard
- **What stands out:** {valuation_trend} trend paired with {volatility_label} volatility.
- **Primary question:** Is the current narrative supportive of further upside given risk levels?
- **Bottom line:** {bottom_line}

---

## Decision Lens
### 1. Market Structure
- Period assessed: {state.period}
- Horizon in focus: {state.horizon_days} days
- Storyline: {valuation_tone}

### 2. Risk Review
- Default read: {risk_summary}
- Notes: {notes_text}

### 3. Fundamental Pulse
- Filing types covered: {fundamental_obj.filing_type if fundamental_obj else 'N/A'}
- Executive summary: {fundamental_obj.executive_summary if fundamental_obj else 'Insufficient filing coverage.'}
- Thesis highlights: {fundamental_obj.investment_thesis if fundamental_obj else 'No official thesis compiled.'}

---

## Supporting Detail
### Sentiment & Narrative
{sentiment_section}

### Valuation Context
{valuation_section}

### Fundamental Highlights
{fundamental_section}

### Recent Headlines
{news_section}

---

## Methodology Notes
- Sequenced multi-agent workflow: market data → sentiment → valuation → fundamentals → risk → writer.
- Market data via yfinance; fallback narratives when data unavailable.
- News ingestion uses Polygon (if configured) otherwise synthetic briefs; sentiment generated via LLM feedback loop.
- Fundamental insights extracted through RAG over ingested 10-K/10-Q filings.
- Risk metrics computed from returns-based analytics (volatility, drawdown, VaR).
"""
    # _ = llm.invoke([SystemMessage(content=WRITER_SYSTEM), HumanMessage(content="draft report")])  # tracing
    
    # Create key findings including valuation and fundamental analysis
    key_findings = [
        "Market action reviewed across price, risk, fundamental, and sentiment dimensions.",
        "Composite takeaway blends model-driven metrics with narrative context."
    ]
    if state.valuation:
        key_findings.append(
            f"Market tone: {state.valuation.price_trend} trend amid {state.valuation.volatility_regime} volatility regime."
        )
    if state.sentiment:
        key_findings.append(
            f"Sentiment skew: {state.sentiment.overall_sentiment} with confidence {state.sentiment.confidence_score:.0%}."
        )
    if state.fundamental:
        key_findings.append(
            f"Fundamental signal: Financial health score {state.fundamental.financial_health_score:.1f}/10 from latest filings."
        )

    report = RiskReport(
        ticker=state.ticker,
        as_of=datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'),
        summary=f"Comprehensive risk and valuation analysis for {state.ticker} with sentiment analysis.",
        key_findings=key_findings,
        metrics_table={
            "annual_vol": state.metrics.annual_vol,
            "max_drawdown": state.metrics.max_drawdown,
            "daily_var_95": state.metrics.daily_var_95,
            "sharpe_like": state.metrics.sharpe_like,
        },
        risk_flags=state.metrics.risk_flags,
        methodology=(
            "Gaussian VaR; log returns; daily OHLC from yfinance; "
            "Valuation with 252-day annualization; LLM sentiment analysis."
        ),
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
        sentiment=state.sentiment,
        valuation=state.valuation,
        fundamental=state.fundamental,
        metrics=state.metrics,
        report=report
    )
    return new_state
