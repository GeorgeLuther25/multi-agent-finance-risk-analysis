import ast
import json
import numpy as np
import pandas as pd
import json
import re
from io import StringIO
from datetime import datetime
from typing import Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from utils.config import get_llm
from utils.tools import get_price_history, get_recent_news, query_10k_documents, period_to_months_range
from utils.constants import RISK_SYSTEM, SENTIMENT_SYSTEM, VALUATION_SYSTEM, FUNDAMENTAL_SYSTEM
from utils.schemas import (
    MarketData, NewsBundle, NewsItem, RiskMetrics, RiskReport,
    SentimentSummary, ValuationMetrics, FundamentalAnalysis, DebateReport
)
from utils.rag_utils import initialize_sample_data, FundamentalRAG, batch_ingest_documents
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv()

# LangSmith visibility
# import os
# os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")


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
    debate: Optional[DebateReport] = None


def parse_agent_response(response_content: str) -> tuple[str, dict]:
    """
    Parse agent response to extract both analysis text and structured data.
    
    Returns:
        tuple: (analysis_text, structured_dict)
    """
    try:
        # Try different markers for structured data
        marker = "STRUCTURED DATA"
        analysis = response_content
        structured_data = {}
        
        # Split by any of the markers
        parts = response_content.split(marker)
        analysis = parts[0].strip()
        if parts and len(parts) > 1:
            # Extract JSON from the second part
            json_part = parts[1]
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', json_part, re.DOTALL)
            json_str = json_match.group(1)
            # Fix common escape issues in JSON
            json_str = json_str.replace('\\$', '$')  # Fix escaped dollar signs
            json_str = json_str.replace("\\'", "'")   # Fix escaped single quotes
            structured_data = json.loads(json_str)
        # else:
        #     print("No structured data marker found, try to find JSON anywhere in the response...")
        #     json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        #     if json_match:
        #         json_str = json_match.group(1)
        #         # Fix common escape issues in JSON
        #         json_str = json_str.replace('\\$', '$')
        #         json_str = json_str.replace("\\'", "'")
        #         structured_data = json.loads(json_str)
        #         # Remove the JSON block from analysis
        #         analysis = re.sub(r'```json\s*\{.*?\}\s*```', '', response_content, flags=re.DOTALL).strip()
        
        return analysis, structured_data
    
    except Exception as e:
        print(f"Error parsing agent response: {e}")
        return response_content, {}


def data_agent(state: State, config: RunnableConfig):
    # llm = get_llm()
    # _ = llm.invoke([SystemMessage(content=DATA_SYSTEM), HumanMessage(content=f"ticker={state.ticker}")])  # no-op, just for tracing
    price_csv = get_price_history.invoke({"ticker": state.ticker, "period": state.period, "interval": state.interval})
    news_raw = get_recent_news.invoke({"ticker": state.ticker, "period": state.period})
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

    Provide a concise summary along with an informed recommendation on whether to invest in this stock for the next {state.horizon_days} days.

    IMPORTANT: Also include the structured json data like the example below, with its values replaced by your analyzed results, and the exact text STRUCTURED DATA.
    STRUCTURED DATA
    ```json
    {{
        "ticker":"...",
        "news_items_analyzed":number of news items here,
        "overall_sentiment":"...",
        "confidence_score":...,
        "investment_recommendation":"...",
        "key_insights":list of key insights of each news item,
        "methodology":"LLM-based reflection-enhanced summarization"
    }}
    ```
    """
    
    sentiment_agent = create_react_agent(llm, [], prompt=(SENTIMENT_SYSTEM))

    # Execute the agent
    try:
        result = sentiment_agent.invoke({"messages": [("human", analysis_prompt)]})
        # Extract the last message content from the result
        if "messages" in result and result["messages"]:
            analysis_content = result["messages"][-1].content
        else:
            analysis_content = "Agent Analysis did not return any message."
    except Exception as e:
        print(f"Agent execution error: {e}")
        analysis_content = f"Tool-based analysis attempted for {state.ticker}"
    
    print(f"analysis_content is: {analysis_content}")
    analysis, structured_data = parse_agent_response(analysis_content)
    if len(structured_data) > 0:
        structured_data["summary"] = analysis_content
        print(f"structured_data type is: {type(structured_data)}. Vlaue is: {structured_data}")
        sentiment_summary = SentimentSummary(**structured_data)
    else:
        print("🚫 No data in parsed structured_data.")
        sentiment_summary = SentimentSummary()
    
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
        from_year, from_month, to_year, to_month = period_to_months_range(state.period)

        # Create agent with tools
        llm = get_llm()
        fundamental_agent = create_react_agent(llm, [query_10k_documents], prompt=(FUNDAMENTAL_SYSTEM))
        analysis_content = ""
        
        # Execute the agent
        try:
            query_msg = (
    f"""
    Please analyze {state.ticker} using the 10-K/10-Q documents from month:{from_month}, year:{from_year} to month:{to_month}, year:{to_year}.
    When calling the query_10k_documents tool, pass your queries as a comma-separated string like this: "financial metrics, business segments, risk factors, competitive position, growth prospects, investment thesis, concerns and risks"
    IMPORTANT: Also include the structured json data like the example below, with its values replaced by your analyzed results, and the exact text STRUCTURED DATA. Ensure the response is valid JSON and keep numeric scores between 0 and 10.
    STRUCTURED DATA
    ```json
    {{
        "executive_summary": "...",
        "key_financial_metrics": {{"metric": "value", "...": "..."}},
        "business_highlights": ["..."],
        "risk_factors": ["..."],
        "competitive_position": "...",
        "growth_prospects": "...",
        "financial_health_score": financial health score here,
        "investment_thesis": "...",
        "concerns_and_risks": ["..."]
    }}
    ```
    """
            )
            result = fundamental_agent.invoke({"messages": [("human", query_msg)]})
            # Extract the last message content from the result
            if "messages" in result and result["messages"]:
                analysis_content = result["messages"][-1].content
            else:
                analysis_content = "Agent Analysis did not return any message."
        except Exception as e:
            print(f"query_10k_documents invocation failed: {e}")

        _, structured_data = parse_agent_response(analysis_content)
        filing_info = available_filings[0]
        if len(structured_data) > 0:
            # Create structured fundamental analysis
            fundamental_analysis = FundamentalAnalysis(
                ticker=state.ticker,
                filing_type=filing_info.get("filing_type", "10-K"),
                filing_date=filing_info.get("ingestion_date", "Unknown"),
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                **structured_data,
                methodology="RAG-enhanced 10-K/10-Q document analysis"
            )
        else:
            print("🚫 No data in parsed structured_data.")
            fundamental_analysis = FundamentalAnalysis()

    
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


def _compute_valuation_metrics(price_csv: str, ticker: str, period: str) -> dict:
    """
    Compute valuation metrics including annualized return and volatility.
    Uses the formulas specified:
    - R_annualized = ((1 + R_cumulative)^(252/n)) - 1
    - σ_annualized = σ_daily × √252
    
    Args:
        price_csv: csv data of stock price
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Period of the observed stock price
        
    Returns:
        Returns dictionary of ValuationMetrics result
    """
    try:
        print(f"📊 Computing valuation metrics for {ticker}...")
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
        
        return {
            "ticker": ticker,
            "analysis_period": period,
            "trading_days": n,
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "daily_volatility": daily_volatility,
            "annualized_volatility": annualized_volatility,
            "price_trend": price_trend,
            "volatility_regime": volatility_regime,
            "valuation_insights": insights,
            "trend_analysis": trend_analysis,
            "risk_assessment": risk_assessment
        }
        
    except Exception as e:
        print(f"Error computing valuation metrics: {e}")
        # Return default metrics on error
        return {
            "ticker": ticker,
            "analysis_period": period,
            "trading_days": 0,
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "daily_volatility": 0.0,
            "annualized_volatility": 0.0,
            "price_trend": "sideways",
            "volatility_regime": "medium",
            "valuation_insights": ["Error in calculation - insufficient data"],
            "trend_analysis": "Unable to determine trend due to data issues",
            "risk_assessment": "Cannot assess risk due to insufficient data"
        }


def valuation_agent(state: State, config: RunnableConfig):
    """
    Valuation agent that analyzes historical price data to compute valuation metrics
    and trends using computational tools for volatility and return calculations.
    """
    llm = get_llm()
    valuation_metrics = None
    
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
        try:
            base_metrics = _compute_valuation_metrics(
                state.market.price_csv,
                state.ticker,
                state.period,
            )
            valuation_metrics = ValuationMetrics(**base_metrics)
        except Exception as err:
            print(f"⚠️  Baseline valuation computation failed: {err}")
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
                valuation_insights=["Unable to compute valuation metrics"],
                trend_analysis="Trend analysis unavailable due to computation error.",
                risk_assessment="Risk profile unavailable.",
            )
            base_metrics = None

        valuation_agent = None
        if valuation_metrics:
            valuation_agent = create_react_agent(llm, [_compute_valuation_metrics], prompt=(VALUATION_SYSTEM))

        if valuation_agent and base_metrics:
            analysis_prompt = f"""
            Compute the valuation metrics for {state.ticker}, price_csv:{state.market.price_csv}, period:{state.period}.
            Provide enhanced trend analysis and investment implications for the next {state.horizon_days} days based on these metrics.
            """

        try:
            tool_results = []
            final_response = None
            for chunk in valuation_agent.stream({"messages": [("human", analysis_prompt)]}):
                # Capture tool calls and results
                for node_name, node_update in chunk.items():
                    messages = node_update.get("messages", [])
                    for message in messages:
                        if hasattr(message, 'content') and "tool_call_id" in str(message):
                            # This is a tool result
                            tool_results.append(message.content)
                        elif hasattr(message, 'content'):
                            # This might be the final response
                            final_response = message.content

            if tool_results:
                try:
                    parsed_metrics = ast.literal_eval(tool_results[-1])
                    valuation_metrics = ValuationMetrics(**parsed_metrics)
                except Exception as parse_err:
                    print(f"⚠️  Unable to parse valuation tool output: {parse_err}")
            if final_response:
                valuation_metrics.trend_analysis = final_response
                print(f"trend_analysis is: {final_response}")
            elif valuation_metrics.trend_analysis:
                valuation_metrics.trend_analysis += "\n\nLLM commentary unavailable; using computed metrics only."
            else:
                valuation_metrics.trend_analysis = "LLM commentary unavailable; using computed metrics only."
        except Exception as e:
            print(f"⚠️  Error in LLM analysis: {e}")
            if valuation_metrics.trend_analysis:
                valuation_metrics.trend_analysis += "\n\nLLM commentary unavailable; using computed metrics only."
            else:
                valuation_metrics.trend_analysis = "LLM commentary unavailable; using computed metrics only."
    
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
    if state.sentiment:
        sentiment_lines = [
            "## Sentiment Analysis",
            f"- **Overall Sentiment**: {state.sentiment.overall_sentiment.title()}",
            f"- **Confidence Score**: {state.sentiment.confidence_score:.1%}",
            f"- **News Items Analyzed**: {state.sentiment.news_items_analyzed}",
            f"- **Investment Recommendation**: {state.sentiment.investment_recommendation}",
        ]
        if state.sentiment.key_insights:
            sentiment_lines.append("")
            sentiment_lines.append("### Key Takeaways")
            sentiment_lines.extend(f"- {insight}" for insight in state.sentiment.key_insights)
        if state.sentiment.summary:
            sentiment_lines.append("")
            sentiment_lines.append("### Summary")
            sentiment_lines.append(state.sentiment.summary)
        sentiment_section = "\n".join(sentiment_lines)
    else:
        sentiment_section = "\n".join([
            "## Sentiment Analysis",
            "No sentiment analysis available - insufficient news data.",
        ])

    if state.valuation:
        valuation_lines = [
            "## Valuation Analysis",
            f"- **Analysis Period**: {state.valuation.analysis_period}",
            f"- **Trading Days Analyzed**: {state.valuation.trading_days}",
            f"- **Cumulative Return**: {state.valuation.cumulative_return:.4f} ({state.valuation.cumulative_return:.2%})",
            f"- **Annualized Return**: {state.valuation.annualized_return:.4f} ({state.valuation.annualized_return:.2%})",
            f"- **Daily Volatility**: {state.valuation.daily_volatility:.6f}",
            f"- **Annualized Volatility**: {state.valuation.annualized_volatility:.4f} ({state.valuation.annualized_volatility:.2%})",
            f"- **Price Trend**: {state.valuation.price_trend.title()}",
            f"- **Volatility Regime**: {state.valuation.volatility_regime.title()}",
        ]
        if state.valuation.valuation_insights:
            valuation_lines.append("")
            valuation_lines.append("### Valuation Insights")
            valuation_lines.extend(f"- {insight}" for insight in state.valuation.valuation_insights)
        if state.valuation.trend_analysis:
            valuation_lines.append("")
            valuation_lines.append("### Trend Analysis")
            valuation_lines.append(state.valuation.trend_analysis)
        if state.valuation.risk_assessment:
            valuation_lines.append("")
            valuation_lines.append("### Risk Assessment")
            valuation_lines.append(state.valuation.risk_assessment)
        valuation_section = "\n".join(valuation_lines)
    else:
        valuation_section = "\n".join([
            "## Valuation Analysis",
            "No valuation analysis available - insufficient market data.",
        ])

    if state.fundamental:
        fundamental_lines = [
            "## Fundamental Analysis",
            f"- **Filing Type**: {state.fundamental.filing_type}",
            f"- **Filing Date**: {state.fundamental.filing_date}",
            f"- **Financial Health Score**: {state.fundamental.financial_health_score:.1f}/10.0",
        ]
        if state.fundamental.executive_summary:
            fundamental_lines.append("")
            fundamental_lines.append("### Executive Summary")
            fundamental_lines.append(state.fundamental.executive_summary)
        if state.fundamental.business_highlights:
            fundamental_lines.append("")
            fundamental_lines.append("### Business Highlights")
            fundamental_lines.extend(f"- {highlight}" for highlight in state.fundamental.business_highlights)
        if state.fundamental.risk_factors:
            fundamental_lines.append("")
            fundamental_lines.append("### Risk Factors")
            fundamental_lines.extend(f"- {risk}" for risk in state.fundamental.risk_factors)
        if state.fundamental.competitive_position:
            fundamental_lines.append("")
            fundamental_lines.append("### Competitive Position")
            fundamental_lines.append(state.fundamental.competitive_position)
        if state.fundamental.growth_prospects:
            fundamental_lines.append("")
            fundamental_lines.append("### Growth Prospects")
            fundamental_lines.append(state.fundamental.growth_prospects)
        if state.fundamental.investment_thesis:
            fundamental_lines.append("")
            fundamental_lines.append("### Investment Thesis")
            fundamental_lines.append(state.fundamental.investment_thesis)
        if state.fundamental.concerns_and_risks:
            fundamental_lines.append("")
            fundamental_lines.append("### Concerns and Risks")
            fundamental_lines.extend(f"- {concern}" for concern in state.fundamental.concerns_and_risks)
        fundamental_section = "\n".join(fundamental_lines)
    else:
        fundamental_section = "\n".join([
            "## Fundamental Analysis",
            "No fundamental analysis available - no 10-K/10-Q data found.",
        ])

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

    debate_section = ""
    if state.debate and getattr(state.debate, "consensus_summary", None):
        debate_section = (
            "## Investment Final Recommendation\n"
            f"{state.debate.consensus_summary}\n"
        )

    report_sections = [
        f"# {state.ticker} Investment Brief",
        f"**Last refreshed:** {current_ts}",
        "",
        "---",
        "",
        "## Executive Dashboard",
        f"- **What stands out:** {valuation_trend} trend paired with {volatility_label} volatility.",
        "- **Primary question:** Is the current narrative supportive of further upside given risk levels?",
        f"- **Bottom line:** {bottom_line}",
        "",
        "---",
        "",
        "## Decision Lens",
        "### 1. Market Structure",
        f"- Period assessed: {state.period}",
        f"- Horizon in focus: {state.horizon_days} days",
        f"- Storyline: {valuation_tone}",
        "",
        "### 2. Risk Review",
        f"- Default read: {risk_summary}",
        f"- Notes: {notes_text}",
        "",
        "### 3. Fundamental Pulse",
        f"- Filing types covered: {fundamental_obj.filing_type if fundamental_obj else 'N/A'}",
        f"- Executive summary: {fundamental_obj.executive_summary if fundamental_obj else 'Insufficient filing coverage.'}",
        f"- Thesis highlights: {fundamental_obj.investment_thesis if fundamental_obj else 'No official thesis compiled.'}",
        "",
        "---",
        "",
        "## Supporting Detail",
        "### Sentiment & Narrative",
        sentiment_section,
        "",
        "### Valuation Context",
        valuation_section,
        "",
        "### Fundamental Highlights",
        fundamental_section,
        "",
        "### Recent Headlines",
        news_section,
        "",
        "---",
        "",
        "## Methodology Notes",
        "- Sequenced multi-agent workflow: market data → sentiment → valuation → fundamentals → risk → writer.",
        "- Market data via yfinance; fallback narratives when data unavailable.",
        "- News ingestion uses Polygon (if configured) otherwise synthetic briefs; sentiment generated via LLM feedback loop.",
        "- Fundamental insights extracted through RAG over ingested 10-K/10-Q filings.",
        "- Risk metrics computed from returns-based analytics (volatility, drawdown, VaR).",
    ]

    if debate_section.strip():
        report_sections.extend([
            "",
            "---",
            "",
            debate_section.strip(),
        ])

    md = "\n".join(report_sections)
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

# Debate
def debate_manager(state: State):
    """Debate Manager control debates"""
    current_agent = 'debate_manager'
    llm = get_llm(temperature=0.5)
    new_state = state.model_copy()

    DEBATE_MANAGER_SYSTEM = f"""
                            You are the Debate Manager coordinating three agents: Fundamental, Sentiment, and Valuation.
                            Your task:
                            - Carefully read the specialized agents arguments.
                            - Analyze agreements, disagreements, and the overall tone.
                            - Identify key evidence and logic from each.
                            - Synthesize these viewpoints into **one concise, reasoned conclusion**.
                            - The conclusion must be objective, actionable, and justified.

                            Your output is final conclusion and must follow this concern:
                            - Highlight points of agreement or conflict.
                            - Note which arguments are stronger or better supported.
                            - Provide a single, coherent conclusion that integrates all perspectives.
                            - If uncertainty remains, explain it clearly.
                            - Be balanced, analytical, and clear about judgement to invest in the stock within the next {state.horizon_days} days.

                            Output format:
                            - Output is consise summary
                            - Based on your summary, give a recommendation to 'buy', 'hold', or 'sell'.
                            """
    # - If the conclusion is not converged, output 'Output: Sentiment' or 'Output: Fundamental' to continue the debate.
    # - If the conclusion is converged, output with format 'Output: Accepted' to end the debate.
    
    # Initialize debate
    if new_state.debate.agent_turn_count is None:
        new_state.debate.agent_turn_count = {agent:0 for agent in new_state.debate.agent_list}

    counts = new_state.debate.agent_turn_count.values()
    counts_list = list(counts)

    if len(set(counts_list)) == 1 and all(c > 0 for c in counts_list):
        print(f'Debate Manager turn-{next(iter(counts))-1}')
        messages = [SystemMessage(content=DEBATE_MANAGER_SYSTEM),]
        
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                messages.append(HumanMessage(content=f"This is based on {agent.title()} Agent arguments: \"\"{latest}\"\""))
                # print(f"\n\n{agent.title()} latest: {latest}")

        response = llm.invoke(messages)
    
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
        prev_consensus_summary = new_state.debate.consensus_summary
        new_state.debate.consensus_summary = response_text
        new_state.debate.agent_arguments[current_agent].append(response_text)
        # print(f"manager conclusions: {response_text}")

        if all(c > 1 for c in counts_list):
            messages = [SystemMessage(content=DEBATE_MANAGER_SYSTEM),]
            messages.append(HumanMessage(content=prev_consensus_summary))
            messages.append(HumanMessage(content=response_text))
            messages.append(HumanMessage(content="""Based on both Consensus Summaries, do this action:
                                                - Compare both Consensus Summaries, and get the final recommendation
                                                - Say in this format 'First: {your first recommendation}, Second: {your second recommendation}, Action: {DEBATE or END}'
                                                - If both Consensus Summaries have similar recommendations, just fill Action with 'END', else than that say 'DEBATE'.
                                                    """))
            response = llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            new_state.debate.terminated = 'END' if response_text.__contains__("END") else ''
    
    #- If you only have one consensus summary, output 'Output: Sentiment' or 'Output: Fundamental' to clarify.
    # - If you have done two concusion before, and your conclusion for positions either 'buy','hold', or 'sell' is changed from your previous conclusion, output 'Output: Sentiment' or 'Output: Fundamental' to continue the debate.
    # - If you have done concusion before, and your conclusion for positions either 'buy','hold', or 'sell' is not changed from your previous conclusion, output with format 'Output: Accepted' to end the debate.
    # print(new_state.debate.agent_turn_count)

    if all(v == new_state.debate.agent_max_turn-1 for v in new_state.debate.agent_turn_count.values()):
        # print("ARGS:", new_state.debate.agent_arguments)
        new_state.debate.terminated = "ENDMAX"

    if new_state.debate.terminated == "END" or new_state.debate.terminated == "ENDMAX":
        messages = [SystemMessage(content=DEBATE_MANAGER_SYSTEM),]
        messages.append(HumanMessage(content=f"""
                                        Refine the Final Consensus into a clear, concise summary suitable for a report.
                                        - Make it brief and precise.
                                        - Do not mention agents or the debate process.
                                        - Output only the final polished summary, with no headings or extra text.

                                        Final Consensus:
                                        "{new_state.debate.consensus_summary}"
                                     """))
        llm.temperature = 0.4
        response = llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        new_state.debate.consensus_summary = response_text
        new_state.debate.agent_arguments[current_agent].append(response_text)
        print(f"Final Consensus Summary:\n{new_state.debate.consensus_summary}")

    return new_state

def route_debate(state: State):
    if len(set(state.debate.agent_turn_count.values())) == 1:
        print(f"\n\nDebate routing turn:{min(state.debate.agent_turn_count.values())} ============================")
    if state.debate.terminated == "ENDMAX" or state.debate.terminated == "END":
        return "END"
    elif min(state.debate.agent_turn_count, key=state.debate.agent_turn_count.get) == "fundamental":
        return "Fundamental"
    elif min(state.debate.agent_turn_count, key=state.debate.agent_turn_count.get) == "sentiment":
        return "Sentiment"
    elif min(state.debate.agent_turn_count, key=state.debate.agent_turn_count.get) == "valuation":
        return "Valuation"

def debate_fundamental_agent(state: State):
    current_agent = 'fundamental'
    llm = get_llm(temperature=0.5)
    new_state = state.model_copy()
    idx = new_state.debate.agent_turn_count[current_agent]
    print(f'{current_agent} turn-{idx}')

    fundamental_summary = ""
    if state.fundamental:
        fundamental_summary = getattr(state.fundamental, "executive_summary", "") or ""
    if not fundamental_summary:
        fundamental_summary = (
            f"No fundamental analysis available for {state.ticker} yet. "
            "Use only your own reasoning and any prior debate context."
        )

    DEBATE_SYSTEM = f"""
                        You are the Fundamental Analysis Agent.

                        Specialization:
                        Focus on financial performance, balance sheet strength, profitability, debt, valuation, and long-term business potential.

                        Report:
                        \"\"\"{fundamental_summary}\"\"\"

                        """
    
    if new_state.debate.agent_turn_count[current_agent] == 0:
        # First Analysis
        initial_analysis_prompt = f"""
                                    Task:
                                    - Make investment recommendation analysis for {state.ticker} within the next {state.horizon_days} days, based ONLY your specialization.\n
                                    - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                                        """
        # DEBATE_SYSTEM += "\n Output a concise summary emphasizing key fundamental insights."
        messages = [SystemMessage(content=DEBATE_SYSTEM),
                    HumanMessage(content=initial_analysis_prompt)
                    ]
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
    else:
        # Debate
        DEBATE_SYSTEM += f"""You will given other Agents judgements, then:
                            - Challenge the judgement from your specialization, don't put heading for this section.\n
                            - Make investment recommendation analysis for {state.ticker} based ONLY your specialization.\n
                            - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                            - If the other Agent provides strong evidence that challenges your recommendation, you may revise your recommendation accordingly.
                         """
        
        messages = [SystemMessage(content=DEBATE_SYSTEM)]
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                if agent != current_agent:
                    messages.append(HumanMessage(content=f"This is judgement based on {agent.title()} Agent for your considerations: \"{latest}\""))
                # print(f"\n\n{agent.title()} latest: {latest}")

        # print(messages)
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)

    new_state.debate.agent_arguments[current_agent].append(response_text)
    new_state.debate.agent_turn_count[current_agent] += 1
    # print(f"result: {response_text}")
    return new_state

def debate_sentiment_agent(state: State):
    current_agent = 'sentiment'
    llm = get_llm(temperature=0.6)
    new_state = state.model_copy()
    idx = new_state.debate.agent_turn_count[current_agent]
    print(f'{current_agent} turn-{idx}')

    DEBATE_SYSTEM = f"""
                        You are the Sentiment Analysis Agent.

                        Specialization:
                        Analyze the tone, emotional language, and implied investor sentiment in a report.
                        Identify whether the sentiment is optimistic, neutral, or negative, and explain why.

                        Report:
                        \"\"\"{state.sentiment.summary}\"\"\"

                        """
    
    if new_state.debate.agent_turn_count[current_agent] == 0:
        # First Analysis
        # initial_analysis_prompt = f"""
        #                             Summarize the overall sentiment:
        #                             - Describe the tone (positive, neutral, or negative)
        #                             - Mention emotional or linguistic indicators of this tone
        #                             - Explain how investors might feel after reading it
        #                                 """
        initial_analysis_prompt = f"""
                                    Task:
                                    - Make investment recommendation analysis for {state.ticker} within the next {state.horizon_days} days, based ONLY your specialization.\n
                                    - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                                  """
        # DEBATE_SYSTEM += "\n Output a concise summary emphasizing key fundamental insights."
        messages = [SystemMessage(content=DEBATE_SYSTEM),
                    HumanMessage(content=initial_analysis_prompt)
                    ]
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
    else:
        # Debate
        DEBATE_SYSTEM += f"""You will given other agents judgements, then:
                            - Challenge the judgement from your specialization, don't put heading for this section..\n
                            - Make investment recommendation analysis for {state.ticker} based ONLY your specialization.\n
                            - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                            - If the other agent provides strong evidence that challenges your recommendation, you may revise your recommendation accordingly.
                         """
        
        messages = [SystemMessage(content=DEBATE_SYSTEM)]
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                if agent != current_agent:
                    messages.append(HumanMessage(content=f"This is judgement based on {agent.title()} Agent for your considerations: \"{latest}\""))
                # print(f"\n\n{agent.title()} latest: {latest}")
    
        # print(messages)
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)

    new_state.debate.agent_arguments[current_agent].append(response_text)
    new_state.debate.agent_turn_count[current_agent] += 1
    # print(f"result: {response_text}")
    return new_state

def debate_valuation_agent(state: State):
    current_agent = 'valuation'
    llm = get_llm(temperature=0.5)
    new_state = state.model_copy()
    idx = new_state.debate.agent_turn_count[current_agent]
    print(f'{current_agent} turn-{idx}')

    DEBATE_SYSTEM = f"""
                        You are the Valuation Analysis Agent.

                        Specialization:
                        Analyze the valuation trends of a given asset or portfolio over an extended time horizon. 
                        To complete the task, you must analyze the historical valuation data of the asset or portfolio provided, identify trends and patterns in valuation metrics over time, and interpret the implications of these trends for investors or stakeholders.

                        Focus your analysis on:
                        1. Price trend analysis (upward, downward, sideways movement)
                        2. Volatility regime assessment (low, medium, high volatility periods)
                        3. Risk-return profile evaluation
                        4. Investment implications and outlook
                        5. Key patterns and inflection points in the data

                        Report:
                        \"\"\"{state.valuation.trend_analysis}\"\"\"

                        """
    
    if new_state.debate.agent_turn_count[current_agent] == 0:
        # First Analysis
        initial_analysis_prompt = f"""
                                    Task:
                                    - Make investment recommendation analysis for {state.ticker} within the next {state.horizon_days} days, based ONLY your specialization.\n
                                    - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                                        """
        # DEBATE_SYSTEM += "\n Output a concise summary emphasizing key fundamental insights."
        messages = [SystemMessage(content=DEBATE_SYSTEM),
                    HumanMessage(content=initial_analysis_prompt)
                    ]
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)
    else:
        # Debate
        DEBATE_SYSTEM += f"""You will given other Agents judgements, then:
                            - Challenge the judgement from your specialization, don't put heading for this section.\n
                            - Make investment recommendation analysis for {state.ticker} based ONLY your specialization.\n
                            - Based on your analysis, give a recommendation to 'buy', 'hold', or 'sell'.
                            - If the other Agent provides strong evidence that challenges your recommendation, you may revise your recommendation accordingly.
                         """
        
        messages = [SystemMessage(content=DEBATE_SYSTEM)]
        for agent in list(new_state.debate.agent_list): #["fundamental", "sentiment"]:
            latest = state.debate.agent_arguments[agent][-1] if state.debate.agent_arguments[agent] else None
            if latest is not None:
                if agent != current_agent:
                    messages.append(HumanMessage(content=f"This is judgement based on {agent.title()} Agent for your considerations: \"{latest}\""))
                # print(f"\n\n{agent.title()} latest: {latest}")

        # print(messages)
        response = llm.invoke(messages)
        # Parse the LLM response to extract structured information
        response_text = response.content if hasattr(response, 'content') else str(response)

    new_state.debate.agent_arguments[current_agent].append(response_text)
    new_state.debate.agent_turn_count[current_agent] += 1
    # print(f"result: {response_text}")
    return new_state
