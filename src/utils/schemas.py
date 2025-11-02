from typing import List, Literal, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class MarketData(BaseModel):
    ticker: str
    period: str
    interval: str
    price_csv: str


class NewsItem(BaseModel):
    date: str
    headline: str
    sentiment: Literal["positive", "neutral", "negative"]
    content: str


class NewsBundle(BaseModel):
    ticker: str
    window_days: int
    items: List[NewsItem] = Field(default_factory=list)


class RiskMetrics(BaseModel):
    ticker: str
    horizon_days: int
    annual_vol: float
    max_drawdown: float
    daily_var_95: float
    sharpe_like: Optional[float] = None
    notes: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)


class SentimentSummary(BaseModel):
    ticker: str = ""
    news_items_analyzed: int = 0
    overall_sentiment: str = ""
    confidence_score: float = 0
    summary: str = ""
    investment_recommendation: str = ""
    key_insights: List[str] = []
    methodology: str = ""


class ValuationMetrics(BaseModel):
    ticker: str
    analysis_period: str
    trading_days: int
    cumulative_return: float
    annualized_return: float
    daily_volatility: float
    annualized_volatility: float
    price_trend: Literal["upward", "downward", "sideways"]
    volatility_regime: Literal["low", "medium", "high"]
    valuation_insights: List[str] = Field(default_factory=list)
    trend_analysis: str
    risk_assessment: str
    methodology: str = "Computational analysis with 252 trading days assumption"


class RiskReport(BaseModel):
    ticker: str
    as_of: str
    summary: str
    key_findings: List[str]
    metrics_table: Dict[str, Any]
    risk_flags: List[str]
    methodology: str
    markdown_report: str


class FundamentalAnalysis(BaseModel):
    ticker: str = ""
    filing_type: str = ""  # "10-K" or "10-Q"
    filing_date: str = ""
    analysis_date: str = ""
    executive_summary: str = ""
    key_financial_metrics: Dict[str, Any] = {}
    business_highlights: List[str] = []
    risk_factors: List[str] = []
    competitive_position: str = ""
    growth_prospects: str = ""
    financial_health_score: float = 0
    investment_thesis: str = ""
    concerns_and_risks: List[str] = []
    methodology: str = "RAG-enhanced 10-K/10-Q document analysis"
