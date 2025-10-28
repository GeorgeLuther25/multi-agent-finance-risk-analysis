"""
Debate-style orchestration for the finance risk analysis agents.

This module mirrors the capabilities of ``src/agents.py`` but introduces
a leader agent that actively debates with each specialist agent.  The
leader consults every agent at least twice before accepting a final
answer, logging a transcript of the interactions along the way.  Once
the leader is satisfied, it synthesises a decision-ready state that can
be consumed by downstream LangGraph workflows.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from pydantic import Field

from agents import (
    State,
    data_agent,
    sentiment_agent,
    valuation_agent,
    fundamental_agent,
    risk_agent,
    writer_agent,
)


AgentFn = Callable[[State, RunnableConfig], State]


class DebateState(State):
    """Extends the standard State with a debate transcript."""

    transcript: List[str] = Field(default_factory=list)


def _ensure_config(config: Optional[RunnableConfig]) -> RunnableConfig:
    return config or RunnableConfig()


def _to_debate_state(source: State, transcript: List[str]) -> DebateState:
    data = source.dict()
    data["transcript"] = transcript
    return DebateState(**data)


def _leader_synthesis(agent_name: str, state: State) -> str:
    if agent_name == "data":
        if state.market and state.market.price_csv:
            return (
                "Leader notes market data retrieved successfully; preparing for sentiment audit."
            )
        return "Leader observes missing market inputs; will cross-check downstream risk."
    if agent_name == "sentiment":
        if state.sentiment:
            return (
                f"Leader reads sentiment as {state.sentiment.overall_sentiment} "
                f"with confidence {state.sentiment.confidence_score:.0%}."
            )
        return "Leader flags lack of sentiment evidence."
    if agent_name == "valuation":
        if state.valuation:
            return (
                f"Leader interprets valuation trend as {state.valuation.price_trend} "
                f"with {state.valuation.volatility_regime} volatility."
            )
        return "Leader unable to form valuation view due to incomplete data."
    if agent_name == "fundamental":
        if state.fundamental:
            return (
                f"Leader assesses financial health score at "
                f"{state.fundamental.financial_health_score:.1f}/10."
            )
        return "Leader notes fundamental evidence absent; recommends manual review."
    if agent_name == "risk":
        if state.metrics:
            flags = ", ".join(state.metrics.risk_flags) if state.metrics.risk_flags else "none"
            return f"Leader evaluates risk flags: {flags}."
        return "Leader lacks computed risk metrics; cautious stance adopted."
    if agent_name == "writer":
        if state.report:
            return "Leader finalised report narrative; ready for dissemination."
        return "Leader could not assemble final report."
    return "Leader awaiting further insight."


def _debate_agent(
    agent_name: str,
    agent_fn: AgentFn,
    state: DebateState,
    config: RunnableConfig,
) -> DebateState:
    transcript = list(state.transcript)
    working_state: State = state

    for round_idx in range(2):
        transcript.append(
            f"[Leader → {agent_name.capitalize()}] Round {round_idx + 1}: "
            "Provide your latest findings based on current evidence."
        )
        updated_state = agent_fn(working_state, config)
        transcript.append(
            f"[{agent_name.capitalize()} → Leader] Round {round_idx + 1}: "
            "Analysis delivered."
        )
        working_state = updated_state

    transcript.append(f"[Leader] Synthesis after {agent_name}: {_leader_synthesis(agent_name, working_state)}")
    return _to_debate_state(working_state, transcript)


def leader_agent(state: DebateState, config: Optional[RunnableConfig] = None) -> DebateState:
    """Primary debate orchestrator that consults every specialist agent twice."""
    cfg = _ensure_config(config)
    debate_state = state

    debate_state = _debate_agent("data", data_agent, debate_state, cfg)
    debate_state = _debate_agent("sentiment", sentiment_agent, debate_state, cfg)
    debate_state = _debate_agent("valuation", valuation_agent, debate_state, cfg)
    debate_state = _debate_agent("fundamental", fundamental_agent, debate_state, cfg)
    debate_state = _debate_agent("risk", risk_agent, debate_state, cfg)
    debate_state = _debate_agent("writer", writer_agent, debate_state, cfg)

    debate_state.transcript.append(
        "[Leader] Debate concluded. Consolidated view ready for decision-makers."
    )
    return debate_state


def build_debate_graph():
    """Compile a LangGraph graph that runs the debate-oriented workflow."""
    graph = StateGraph(DebateState)
    graph.add_node("leader", leader_agent)
    graph.set_entry_point("leader")
    graph.add_edge("leader", END)
    return graph.compile()


def run_debate(initial_state: DebateState, config: Optional[RunnableConfig] = None) -> DebateState:
    """
    Convenience helper to execute the debate loop without constructing a graph manually.
    """
    return leader_agent(initial_state, config)
