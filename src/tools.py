import pandas as pd
import yfinance as yf
import requests
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
from langchain.tools import tool
from textblob import TextBlob
import warnings
warnings.filterwarnings("ignore")

# Add web scraping capabilities for URL content extraction
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("⚠️  BeautifulSoup not available. Install with: pip install beautifulsoup4")

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("⚠️  newspaper3k not available. Install with: pip install newspaper3k")

@tool("get_price_history")
def get_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Returns price history CSV for ticker using yfinance."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        return f"ERROR: No data for {ticker}."
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Clean up column names - handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex columns and remove ticker prefixes
        df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] for col in df.columns]
    else:
        # Handle regular columns with ticker prefixes
        df.columns = [col.replace(f"{ticker},", "").strip() if isinstance(col, str) else col for col in df.columns]
    
    # Ensure Date column is properly formatted
    if 'Date' in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    
    return df.to_csv(index=False)

@tool("get_recent_news")
def get_recent_news(ticker: str, days: int = 14) -> str:
    # """Stub news. Replace with your provider later."""
    # cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    # samples = [
    #     {"date": cutoff, "headline": f"{ticker} announces product update", "sentiment": "positive"},
    #     {"date": cutoff, "headline": f"{ticker} faces regulatory query", "sentiment": "negative"},
    # ]
    # return str(samples)
    """
    Retrieves recent financial news for a given ticker using Polygon.io.
    Returns a list of news items with headlines, content, dates, and sentiment analysis.
    """
    import os
    
    polygon_key = os.getenv("POLYGON_API_KEY")
    if not polygon_key:
        print("⚠️  POLYGON_API_KEY not found. Using fallback synthetic news.")
        print("   Get your free API key at: https://polygon.io/")
        print("   Set it with: export POLYGON_API_KEY='your-key-here'")
        # Fallback to synthetic news
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return str(_generate_synthetic_news(ticker, cutoff_date))
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        news_items = _get_polygon_news_with_content(ticker, polygon_key, cutoff_date)
        
        if not news_items:
            print(f"⚠️  No recent news found for {ticker} from Polygon.io. Using fallback.")
            news_items = _generate_synthetic_news(ticker, cutoff_date)
        
        # Sort by date (most recent first) and limit results
        news_items: List[Dict[str, Any]] = sorted(news_items, key=lambda x: x['date'], reverse=True)[:8]
        # print("News items are:", news_items)
        return str(news_items)
        # return news_items
        
    except Exception as e:
        print(f"❌ Error fetching news for {ticker}: {e}")
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return str(_generate_synthetic_news(ticker, cutoff_date))


def _extract_url_content(url: str, max_length: int = 4000) -> str:
    """
    Extract full article content from a URL using multiple methods.
    Returns the full article text or empty string if extraction fails.
    """
    if not url or not url.startswith(('http://', 'https://')):
        return ""
    
    print(f"🔍 Extracting content from URL: {url[:70]}...")
    
    # Method 1: Try newspaper3k (best for news articles)
    if NEWSPAPER_AVAILABLE:
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            content = article.text.strip()
            if content and len(content) > 300:  # Minimum substantial content threshold
                # Clean and truncate content
                content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
                content = content[:max_length] + "..." if len(content) > max_length else content
                print(f"✅ Extracted {len(content)} characters using newspaper3k")
                return content
        except Exception as e:
            print(f"⚠️  newspaper3k extraction failed: {e}")
    
    # Method 2: Try BeautifulSoup as fallback
    if BEAUTIFULSOUP_AVAILABLE:
        try:
            # Use a realistic user agent to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'iframe']):
                    element.decompose()
                
                # Try different content selectors in order of preference
                content_selectors = [
                    'article',
                    '.article-content',
                    '.entry-content',
                    '.post-content',
                    '.story-body',
                    '.article-body',
                    '.content',
                    '[data-module="ArticleBody"]',
                    '.text-content',
                    'main',
                    '.article-text'
                ]
                
                content = ""
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        content = ' '.join([elem.get_text().strip() for elem in elements])
                        if len(content) > 300:  # Found substantial content
                            break
                
                # Fallback: Extract from paragraphs
                if not content or len(content) < 300:
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 30])
                
                if content and len(content) > 300:
                    # Clean content
                    content = re.sub(r'\s+', ' ', content)
                    
                    # Remove common junk text patterns
                    junk_patterns = [
                        r'.*cookies.*accept.*',
                        r'.*subscribe.*newsletter.*',
                        r'.*follow us.*social.*',
                        r'.*privacy policy.*',
                        r'.*terms of service.*',
                        r'.*advertisement.*',
                        r'.*ad\s+choices.*'
                    ]
                    for pattern in junk_patterns:
                        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
                    
                    content = content[:max_length] + "..." if len(content) > max_length else content
                    print(f"✅ Extracted {len(content)} characters using BeautifulSoup")
                    return content
                    
        except Exception as e:
            print(f"⚠️  BeautifulSoup extraction failed: {e}")
    
    # Method 3: Simple text extraction fallback
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Very basic text extraction
            text = response.text
            # Remove HTML tags with simple regex
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) > 500:
                clean_text = clean_text[:max_length] + "..." if len(clean_text) > max_length else clean_text
                print(f"✅ Extracted {len(clean_text)} characters using simple extraction")
                return clean_text
                
    except Exception as e:
        print(f"⚠️  Simple extraction failed: {e}")
    
    print(f"❌ Could not extract content from {url}")
    return ""


def _get_polygon_news_with_content(ticker: str, api_key: str, cutoff_date: datetime) -> List[Dict[str, Any]]:
    """
    Get recent news from Polygon.io with enhanced content extraction.
    Now extracts full article content from URLs when available.
    """
    try:
        print(f"� Fetching Polygon.io news for {ticker}...")
        # Format date for Polygon API (YYYY-MM-DD)
        from_date = cutoff_date.strftime('%Y-%m-%d')
        
        # Construct API request
        base_url = "https://api.polygon.io/v2/reference/news"
        params = {
            'ticker': ticker,
            'published_utc.gte': from_date,
            'order': 'desc',
            'limit': 10,
            'apikey': api_key
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                news_items = []
                
                for article in data['results'][:5]:  # Limit to 5 articles for processing
                    try:
                        # Extract basic info
                        title = article.get('title', 'No title')
                        description = article.get('description', '')
                        article_url = article.get('article_url', '')
                        published_utc = article.get('published_utc', '')
                        author = article.get('author', 'Unknown')
                        publisher = article.get('publisher', {}).get('name', 'Unknown')
                        
                        # Skip if no title
                        if not title or len(title) < 10:
                            continue
                        
                        # Try to get full content from URL
                        full_content = ""
                        if article_url:
                            full_content = _extract_url_content(article_url, max_length=3000)
                        
                        # Use full content if available, otherwise fall back to description
                        content = full_content if full_content else description
                        
                        # Only include if we have substantial content
                        if content and len(content) > 100:
                            # Enhance content with keywords if available
                            if 'keywords' in article and article['keywords']:
                                keywords = ', '.join(article['keywords'][:5])
                                content += f"\n\nKey topics: {keywords}"
                            
                            # Analyze sentiment on full text
                            full_text = f"{title} {content}"
                            sentiment = _analyze_sentiment(full_text)
                            
                            # Parse date
                            if published_utc:
                                news_date = datetime.fromisoformat(published_utc.replace('Z', '+00:00'))
                            else:
                                news_date = datetime.now()
                            
                            news_item = {
                                'date': news_date.strftime('%Y-%m-%d'),
                                'headline': title,
                                'content': content,
                                'sentiment': sentiment,
                                'source': f'Polygon.io ({publisher})',
                                'author': author,
                                'url': article_url,
                                'ticker': ticker,
                                'extracted_full_content': bool(full_content)  # Track if we got full content
                            }
                            news_items.append(news_item)
                            
                    except Exception as e:
                        print(f"⚠️  Error processing Polygon article: {e}")
                        continue
                
                print(f"✅ Retrieved {len(news_items)} Polygon.io articles with enhanced content")
                return news_items
            else:
                print("⚠️  No news results found in Polygon.io response")
                return []
        else:
            print(f"❌ Polygon.io API error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Polygon.io news error: {e}")
        return []


def _analyze_sentiment(text: str) -> str:
    # """Simple sentiment analysis using keyword matching and basic NLP"""
    # if not text:
    #     return "neutral"
    
    # text_lower = text.lower()
    
    # # Positive indicators
    # positive_words = [
    #     'gains', 'growth', 'profit', 'earnings beat', 'upgrade', 'bullish', 'positive',
    #     'strong', 'revenue', 'success', 'expansion', 'innovation', 'partnership',
    #     'acquisition', 'dividend', 'buyback', 'outperform', 'record', 'soars'
    # ]
    
    # # Negative indicators
    # negative_words = [
    #     'loss', 'decline', 'falls', 'drops', 'downgrade', 'bearish', 'negative',
    #     'weak', 'miss', 'concern', 'investigation', 'lawsuit', 'regulatory',
    #     'warning', 'cut', 'layoff', 'bankruptcy', 'plunges', 'crash'
    # ]
    
    # positive_count = sum(1 for word in positive_words if word in text_lower)
    # negative_count = sum(1 for word in negative_words if word in text_lower)
    
    # if positive_count > negative_count:
    #     return "positive"
    # elif negative_count > positive_count:
    #     return "negative"
    # else:
    #     return "neutral"
    return "neutral"


def _generate_synthetic_news(ticker: str, cutoff_date: datetime) -> List[Dict[str, Any]]:
    """Stub news. Replace with your provider later."""
    cutoff = cutoff_date.strftime("%Y-%m-%d")
    samples = [
        {"date": cutoff, "headline": f"{ticker} announces product update", "sentiment": "positive"},
        {"date": cutoff, "headline": f"{ticker} faces regulatory query", "sentiment": "negative"},
    ]
    return str(samples)


@tool("query_10k_documents")
def query_10k_documents(ticker: str, query: str) -> str:
    """
    Query 10-K/10-Q documents using RAG (Retrieval-Augmented Generation).
    
    Use this tool to extract specific information from 10-K/10-Q filings.
    Provide a natural language query about the company's fundamentals.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        query: Natural language question about the company's 10-K filing
               (e.g., "What are the key financial metrics?", "What are the main risk factors?")
    
    Returns:
        Relevant information extracted from the 10-K/10-Q documents
    """
    from .rag_utils import FundamentalRAG, initialize_sample_data
    
    try:
        # Initialize RAG system
        rag_system = FundamentalRAG()
        
        # Check if we have data for this ticker, if not initialize sample data
        available_filings = rag_system.get_available_filings(ticker.upper())
        if not available_filings:
            print(f"No filings found for {ticker}, initializing sample data...")
            initialize_sample_data(rag_system)
            available_filings = rag_system.get_available_filings(ticker.upper())
        
        if not available_filings:
            return f"No 10-K/10-Q data available for {ticker}. Unable to provide fundamental analysis."
        
        # Retrieve relevant document chunks
        relevant_chunks = rag_system.retrieve_relevant_chunks(ticker.upper(), query, k=5)
        
        if not relevant_chunks:
            return f"No relevant information found for query: '{query}' in {ticker}'s filings."
        
        # Combine chunk content
        context = "\n\n---DOCUMENT SECTION---\n".join([
            chunk.page_content for chunk in relevant_chunks
        ])
        
        # Return the context for the LLM to process
        return f"Retrieved information from {ticker}'s 10-K filing:\n\n{context}"
        
    except Exception as e:
        return f"Error querying 10-K documents for {ticker}: {str(e)}"