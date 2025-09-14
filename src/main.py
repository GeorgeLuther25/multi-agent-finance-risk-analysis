import os
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

# LangSmith (optional) - disabled to avoid API key warnings
# os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")

from .agents import (
    State, data_agent, risk_agent, sentiment_agent,
    valuation_agent, fundamental_agent, writer_agent
)
# from .agents import State, data_agent, risk_agent, writer_agent

def build_graph():
    g = StateGraph(State)
    g.add_node("data", data_agent)
    g.add_node("sentiment", sentiment_agent)
    g.add_node("valuation", valuation_agent)
    g.add_node("fundamental", fundamental_agent)
    g.add_node("risk", risk_agent)
    g.add_node("writer", writer_agent)

    g.set_entry_point("data")
    g.add_edge("data", "sentiment")
    g.add_edge("sentiment", "valuation")
    g.add_edge("valuation", "fundamental")
    g.add_edge("fundamental", "risk")
    g.add_edge("risk", "writer")
    # g.add_edge("data", END)
    g.add_edge("writer", END)
    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    state = State(ticker="AAPL", period="1wk", interval="1d", horizon_days=30)
    final_state = graph.invoke(state, config=RunnableConfig())
    
    # Handle the return value properly
    if isinstance(final_state, dict):
        # If final_state is a dict, access it as such
        if 'report' in final_state and final_state['report']:
            print(final_state['report'].markdown_report)
        else:
            print("Final state:", final_state)
    else:
    # If final_state is a State object
        if hasattr(final_state, 'report') and final_state.report:
            print(final_state.report.markdown_report)
        # else:
    print("Final state:", final_state)
    if hasattr(final_state, 'metrics'):
        print("Metrics:", final_state.metrics)
    if hasattr(final_state, 'market'):
        print("Market data available:", final_state.market is not None)
    if hasattr(final_state, 'sentiment'):
        print("Sentiment available:", final_state.sentiment is not None)
