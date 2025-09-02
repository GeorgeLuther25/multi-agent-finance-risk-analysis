import streamlit as st
from openai import OpenAI
import os
import yfinance as yf
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta

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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("---")
user_input = st.text_area("Your message:", value="", height=100, key="user_input")
submit = st.button("Send", type="primary")

if submit and user_input.strip():
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        try:
            # Check if user is asking about stocks and automatically fetch data
            if any(keyword in user_input.lower() for keyword in ['stock', 'ticker', 'price', 'chart', 'aapl', 'tsla', 'googl', 'msft', 'amzn', 'nvda', 'meta', 'nflx']):
                # Try to extract stock symbol
                import re
                symbols = re.findall(r'\b[A-Z]{1,5}\b', user_input.upper())
                
                if symbols:
                    symbol = symbols[0]
                    # Get stock data and show chart
                    stock_data = get_stock_data(symbol)
                    
                    if "error" not in stock_data:
                        # Display stock chart
                        chart = create_stock_chart(symbol)
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Add stock data context to the conversation
                        context = (f"Current stock data for {symbol}: "
                                 f"Price: ${stock_data['current_price']}, "
                                 f"Change: {stock_data['change']} ({stock_data['change_percent']}%), "
                                 f"Volume: {stock_data['volume']}")
                        enhanced_prompt = f"{user_input}\n\nCurrent real-time data: {context}"
                        st.session_state.chat_history[-1]["content"] = enhanced_prompt
            
            # Make API call to GPT-4
            response = client.chat.completions.create(
                model="gpt-5",
                messages=st.session_state.chat_history,
                # temperature=0.7,
                # max_tokens=1000,
            )
            
            assistant_reply = response.choices[0].message.content
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
