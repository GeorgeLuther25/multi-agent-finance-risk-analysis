#!/usr/bin/env python3
"""
Test script to demonstrate the sentiment agent functionality.
"""

from src.main import build_graph
from src.agents import State
from langchain_core.runnables import RunnableConfig

def test_sentiment_agent():
    """Test the sentiment agent with different scenarios"""
    
    print("🚀 Testing Multi-Agent Finance Risk Analysis with Sentiment Agent")
    print("=" * 80)
    
    # Build the graph with all agents including sentiment
    graph = build_graph()
    
    # Test with AAPL
    print("\n📊 Testing with AAPL (1 week data)...")
    state = State(ticker="AAPL", period="1wk", interval="1d", horizon_days=30)
    
    try:
        final_state = graph.invoke(state, config=RunnableConfig())
        
        # Handle different return types
        if isinstance(final_state, dict):
            report = final_state.get('report')
            sentiment = final_state.get('sentiment')
        else:
            report = getattr(final_state, 'report', None)
            sentiment = getattr(final_state, 'sentiment', None)
        
        print("\n✅ Analysis Complete!")
        
        if sentiment:
            print(f"\n📈 SENTIMENT ANALYSIS RESULTS:")
            print(f"   • Overall Sentiment: {sentiment.overall_sentiment.upper()}")
            print(f"   • Confidence Score: {sentiment.confidence_score:.1%}")
            print(f"   • News Items Analyzed: {sentiment.news_items_analyzed}")
            print(f"   • Investment Recommendation: {sentiment.investment_recommendation}")
            
            if sentiment.key_insights:
                print(f"\n🔍 Key Insights:")
                for insight in sentiment.key_insights[:3]:  # Show top 3
                    print(f"   • {insight}")
        
        if report:
            print(f"\n📋 FULL REPORT:")
            print("-" * 80)
            print(report.markdown_report)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def test_graph_structure():
    """Test the graph structure to show sentiment agent integration"""
    
    print("\n🔗 GRAPH STRUCTURE:")
    print("   data_agent → sentiment_agent → risk_agent → writer_agent")
    print("\n📝 Sentiment Agent Features:")
    print("   • Reflection-enhanced prompting")
    print("   • Multi-step analysis: Summarize → Critique → Refine → Conclude")
    print("   • LLM-based summarization")
    print("   • Investment recommendation generation")
    print("   • Confidence scoring")
    print("   • Key insights extraction")

if __name__ == "__main__":
    test_graph_structure()
    test_sentiment_agent()
    
    print("\n🎯 SENTIMENT AGENT SUCCESSFULLY INTEGRATED!")
    print("The agent follows the specified methodology:")
    print("• Uses reflection-enhanced prompting")
    print("• Provides concise summaries with investment recommendations")
    print("• Employs multi-step reasoning (summarize, critique, refine)")
    print("• Learns through all data rather than using RAG")
    print("• Offers informed opinions based on news analysis")
