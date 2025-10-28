from __future__ import annotations

import os
from typing import Optional, Tuple, Type

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from agents import (
    State,
    data_agent,
    risk_agent,
    sentiment_agent,
    valuation_agent,
    fundamental_agent,
    writer_agent,
)


def build_chain_graph():
    """Default linear pipeline graph identical to the original implementation."""
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
    g.add_edge("writer", END)
    return g.compile()


def resolve_mode(mode: Optional[str] = None) -> str:
    env_value = os.getenv("ANALYSIS_MODE")
    value = mode or env_value or "chain"
    value = value.strip().lower()
    if value not in {"chain", "debate"}:
        return "chain"
    return value


def get_workflow(mode: Optional[str] = None) -> Tuple[object, Type[State]]:
    """
    Returns a compiled LangGraph graph and the associated state class based
    on the chosen analysis mode.
    """
    resolved = resolve_mode(mode)
    if resolved == "debate":
        from debate_agents import DebateState, build_debate_graph

        return build_debate_graph(), DebateState
    return build_chain_graph(), State


def build_graph():
    """Backward-compatible helper returning the chain graph."""
    return build_chain_graph()


if __name__ == "__main__":
    graph, state_cls = get_workflow()
    state = state_cls(ticker="AAPL", period="1wk", interval="1d", horizon_days=30)
    final_state = graph.invoke(state, config=RunnableConfig())

    if isinstance(final_state, dict):
        if final_state.get("report"):
            print(final_state["report"].markdown_report)
        else:
            print("Final state:", final_state)
    else:
        if hasattr(final_state, "report") and final_state.report:
            print(final_state.report.markdown_report)
    if hasattr(final_state, "metrics"):
        print("Metrics:", final_state.metrics)
    if hasattr(final_state, "market"):
        print("Market data available:", final_state.market is not None)
    if hasattr(final_state, "sentiment"):
        print("Sentiment available:", final_state.sentiment is not None)
