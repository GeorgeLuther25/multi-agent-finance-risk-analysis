import os
import sys
import json
import time

from src.main import run_all_graphs
from src.utils.config import get_llm


llm = get_llm()


def run_all_graphs_and_save(ticker, period, interval, horizon_days, end_date):
    state_obj = run_all_graphs(ticker, period, interval, horizon_days, end_date)
    json_data = state_obj.model_dump()
    # print(f"The data is: {json_data["debate"]["consensus_summary"]}")
    # consensus = "Alphabet (GOOGL) shows strong financial performance and AI leadership, but faces significant regulatory and competitive risks that could aﬀect long-term profitability. The stock's high volatility suggests potential price fluctuations, necessitating caution. While there is optimism around AI advancements and institutional confidence, these are tempered by the highlighted risks. Short-term gains are possible due to recent price trends, but long-term uncertainties remain. The recommendation is to 'Hold', balancing current strengths with vigilance towards potential risks."
    consensus = json_data["debate"]["consensus_summary"]
    response = llm.invoke(f"Output as a single word-BUY,SELL, or HOLD from the following text: {consensus}")  # tracing
    recommendation = response.content if hasattr(response, 'content') else str(response)
    json_data["debate"]["recommendation"] = recommendation
    json_data["end_date"] = end_date
    with open(f"src/tests/json/final_state_{state_obj.ticker}_{state_obj.period}_{end_date}.json", "w") as f:
        json.dump(json_data, f, indent=2)
    time.sleep(4)  # For giving LLM some time to rest



print("🚀 Running setup before all tests...")

# run_all_graphs_and_save(ticker="GOOGL", period="1wk", interval="1d", horizon_days=30, end_date="2025-3-1")
run_all_graphs_and_save("GOOGL", "1mo", "1d", 30, "2025-4-1")
run_all_graphs_and_save("GOOGL", "1mo", "1d", 30, "2025-6-1")
run_all_graphs_and_save("AAPL", "1mo", "1d", 30, "2025-6-1")
run_all_graphs_and_save("AAPL", "1mo", "1d", 30, "2023-6-1")
run_all_graphs_and_save("TSLA", "1mo", "1d", 30, "2025-1-30")

print("✅ Setup completed!")
