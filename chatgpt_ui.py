import streamlit as st
from openai import OpenAI
import os
import yfinance as yf
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

st.set_page_config(page_title="ChatGPT UI", page_icon="🤖", layout="centered")

st.title("ChatGPT-like UI with OpenAI API")

# api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
# if api_key:
#     openai.api_key = api_key
# else:
#     st.warning("Please enter your OpenAI API key in the sidebar.")
#     st.stop()

api_key = 'add-your-api-key-here'
client = OpenAI(api_key=api_key)

def get_stock_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        
        current_price = data['Close'].iloc[-1]
        prev_close = data['Close'].iloc[-2]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "volume": int(data['Volume'].iloc[-1]),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "52_week_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52_week_low": info.get('fiftyTwoWeekLow', 'N/A')
        }
    except Exception as e:
        return {"error": f"Could not fetch data for {symbol}: {str(e)}"}

def create_stock_chart(symbol, period="1y"):
    """Create a stock chart using plotly"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template="plotly_white"
        )
        
        return fig
    except Exception as e:
        return None


def search_web(query, num_results=3):
    """Search the web for current information"""
    try:
        # Use DuckDuckGo search (no API key required)
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.find_all('a', {'class': 'result__a'})[:num_results]:
            title = result.get_text()
            link = result.get('href')
            if title and link:
                results.append({"title": title, "link": link})
        
        return {"results": results, "query": query}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def get_webpage_content(url):
    """Get content from a webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        text = ' '.join(line for line in lines if line)
        
        return {"content": text[:2000], "url": url}  # Limit content
    except Exception as e:
        return {"error": f"Failed to get content: {str(e)}"}


def analyze_data(data_description, analysis_type="summary"):
    """Analyze data and create visualizations"""
    try:
        # Generate sample data based on description
        if "stock" in data_description.lower():
            # Generate sample stock data
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            
            df = pd.DataFrame({
                'Date': dates,
                'Price': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            })
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Price chart
            ax1.plot(df['Date'], df['Price'])
            ax1.set_title('Stock Price Over Time')
            ax1.set_ylabel('Price ($)')
            
            # Volume chart
            ax2.bar(df['Date'], df['Volume'], alpha=0.7)
            ax2.set_title('Trading Volume')
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "analysis": f"Generated analysis for: {data_description}",
                "chart": image_base64,
                "summary": df.describe().to_dict()
            }
        
        return {"analysis": f"Analysis completed for: {data_description}"}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


# Enhanced function definitions for OpenAI
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_data",
            "description": "Get real-time stock data including current price, changes, volume, and key financial metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., AAPL, TSLA, GOOGL, MSFT)"
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y",
                        "default": "1y"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_web",
            "description": "Search the web for current information, news, or recent developments",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find current information"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of search results to return (1-5)",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_webpage_content", 
            "description": "Get content from a specific webpage URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the webpage to fetch content from"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Analyze data and create visualizations, similar to code interpreter",
            "parameters": {
                "type": "object", 
                "properties": {
                    "data_description": {
                        "type": "string",
                        "description": "Description of data to analyze or visualization to create"
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis: summary, visualization, statistical",
                        "default": "summary"
                    }
                },
                "required": ["data_description"]
            }
        }
    }
]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
user_input = st.text_area("Your message:", value="", height=100, key="user_input")
submit = st.button("Send", type="primary")

if submit and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        try:
            # Enhanced system prompt for ChatGPT-like behavior
            system_prompt = {
                "role": "system",
                "content": """You are ChatGPT, a helpful AI assistant. You have access to several tools:
                - get_stock_data: Get real-time stock information
                - search_web: Search the internet for current information
                - get_webpage_content: Read content from specific webpages
                - analyze_data: Analyze data and create visualizations
                
                Use these tools when users ask about current events, stock prices, data analysis, or need real-time information. 
                Be conversational and helpful, and explain your reasoning when using tools."""
            }
            
            # Add system prompt if not present
            messages = st.session_state.chat_history.copy()
            if not messages or messages[0]["role"] != "system":
                messages = [system_prompt] + messages
            
            # First API call with function calling
            response = client.chat.completions.create(
                # model="gpt-4-turbo-preview",
                model="gpt-5",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                # temperature=0.7,
                # max_tokens=2000,
            )
            
            message = response.choices[0].message
            
            # Check if the model wants to call functions
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Add the assistant message with tool calls
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": message.content,
                    "tool_calls": [{"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
                })
                
                # Execute each function call
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    st.info(f"🔧 Using {function_name} with: {function_args}")
                    
                    # Call the appropriate function
                    if function_name == "get_stock_data":
                        function_result = get_stock_data(**function_args)
                        # Display stock chart if data is available
                        if "error" not in function_result:
                            symbol = function_args.get("symbol", "")
                            chart = create_stock_chart(symbol, function_args.get("period", "1y"))
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                                
                    elif function_name == "search_web":
                        function_result = search_web(**function_args)
                        
                    elif function_name == "get_webpage_content":
                        function_result = get_webpage_content(**function_args)
                        
                    elif function_name == "analyze_data":
                        function_result = analyze_data(**function_args)
                        # Display chart if generated
                        if "chart" in function_result:
                            st.image(base64.b64decode(function_result["chart"]), caption="Generated Analysis")
                    
                    # Add function result to conversation
                    st.session_state.chat_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(function_result)
                    })
                
                # Update messages for final response
                messages = [system_prompt] + st.session_state.chat_history
                
                # Second API call to get the final response
                final_response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                )
                
                assistant_reply = final_response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
            else:
                # No function call needed
                assistant_reply = message.content
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
                
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("## ChatGPT Output")
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    elif msg["role"] == "assistant" and msg.get("content"):
        st.markdown(f"**ChatGPT:** {msg['content']}")
        
# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()
    st.session_state.chat_history = []
    st.rerun()
