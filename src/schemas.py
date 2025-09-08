from typing import List, Literal, Dict, Any, Optional
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
    ticker: str
    news_items_analyzed: int
    overall_sentiment: Literal["bullish", "bearish", "neutral"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    summary: str
    investment_recommendation: str
    key_insights: List[str] = Field(default_factory=list)
    methodology: str = "LLM-based reflection-enhanced summarization"


class RiskReport(BaseModel):
    ticker: str
    as_of: str
    summary: str
    key_findings: List[str]
    metrics_table: Dict[str, Any]
    risk_flags: List[str]
    methodology: str
    markdown_report: str
