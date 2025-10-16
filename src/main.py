import os
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from markdown_pdf import MarkdownPdf, Section
pdf = MarkdownPdf(toc_level=2, optimize=True)

from dotenv import load_dotenv
load_dotenv()

# LangSmith (optional) - disabled to avoid API key warnings
# os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
# os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")

from .agents import (
    State, data_agent, risk_agent, sentiment_agent,
    valuation_agent, fundamental_agent, writer_agent
)
# from .agents import State, data_agent, risk_agent, writer_agent
from .agents import debate_sentiment_agent, debate_valuation_agent, debate_fundamental_agent, debate_manager, route_debate
from .schemas import DebateReport

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

def build_debate_graph():
    g = StateGraph(State)
    # Multi-agent Debate
    g.add_node("debate_manager", debate_manager)
    g.add_node("debate_fundamental", debate_fundamental_agent)
    g.add_node("debate_sentiment", debate_sentiment_agent)
    g.add_node("debate_valuation", debate_valuation_agent)
    g.add_node("writer", writer_agent)
    g.set_entry_point("debate_manager")

    g.add_edge("debate_fundamental", "debate_manager")
    g.add_edge("debate_sentiment", "debate_manager")
    g.add_edge("debate_valuation", "debate_manager")

    g.add_conditional_edges(
        "debate_manager",
        route_debate,
        { 
            "END": "writer",
            "Fundamental":"debate_fundamental",
            "Sentiment":"debate_sentiment",
            "Valuation":"debate_valuation"
        },
    )
    g.add_edge("writer", END)
    
    return g.compile()


if __name__ == "__main__":
    if not os.path.exists("final_state.json"):
        graph = build_graph()
        state = State(ticker="AAPL", period="1wk", interval="1d", horizon_days=30)
        final_state = graph.invoke(state, config=RunnableConfig())
        
        # Handle the return value properly
        # if isinstance(final_state, dict):
        #     # If final_state is a dict, access it as such
        #     if 'report' in final_state and final_state['report']:
        #         print(final_state['report'].markdown_report)
        #     else:
        #         print("Final state:", final_state)
        # else:
        # # If final_state is a State object
        #     if hasattr(final_state, 'report') and final_state.report:
        #         print(final_state.report.markdown_report)
            # else:
        # print("Final state:", final_state)
        # if hasattr(final_state, 'metrics'):
        #     print("Metrics:", final_state.metrics)
        # if hasattr(final_state, 'market'):
        #     print("Market data available:", final_state.market is not None)
        # if hasattr(final_state, 'sentiment'):
        #     print("Sentiment available:", final_state.sentiment is not None)
        print(final_state['report'].markdown_report)

        # Save to JSON file
        final_state_dict = final_state 
        state_obj = State(**final_state_dict)
        with open("final_state.json", "w") as f:
                f.write(state_obj.model_dump_json(indent=2))

    with open("final_state.json", "r") as f:
        final_state = State.model_validate_json(f.read())

        debateGraph = build_debate_graph()
        debateReport = DebateReport(agent_list=["fundamental", "sentiment", "valuation"])
        debateReport.agent_max_turn = 5
        final_state.debate = debateReport
        with open("graph_debate.png", "wb") as f:
            f.write(debateGraph.get_graph().draw_mermaid_png())
        final_state = debateGraph.invoke(final_state, config=RunnableConfig(recursion_limit=100), verbose=True)
        print('Debate Terminated: ',final_state['debate'].terminated)

        pdf.add_section(Section(final_state['report'].markdown_report))
        pdf.save(f"AnalysisReport_{final_state['ticker']}.pdf")

        # Save to JSON file
        final_state_dict = final_state 
        state_obj = State(**final_state_dict)
        with open("final_state_with_debate.json", "w") as f:
                f.write(state_obj.model_dump_json(indent=2))

    # debateGraph = build_debate_graph(final_state)
    # final_state = debateGraph.invoke(final_state, config=RunnableConfig(), verbose=True)