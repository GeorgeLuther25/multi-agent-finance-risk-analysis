from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
import threading
import queue
import requests
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the main module and agents
import main
from agents import State

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['ticker', 'period', 'interval', 'horizon_days']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create state object
        state = State(
            ticker=data['ticker'].upper(),
            period=data['period'],
            interval=data['interval'],
            horizon_days=int(data['horizon_days'])
        )
        
        # Quick Ollama reachability check (fail fast)
        try:
            _ = requests.get(
                'http://localhost:11434/api/tags', timeout=2
            )
        except Exception:
            return jsonify({'error': 'Ollama is not reachable. Start it with: ollama serve'}), 503

        # Build the graph
        graph = main.build_graph()

        # Run with a watchdog timeout so UI won't hang forever
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)

        def _run_graph():
            try:
                res = graph.invoke(state)
                result_queue.put((True, res))
            except Exception as e:
                result_queue.put((False, e))

        worker = threading.Thread(target=_run_graph, daemon=True)
        worker.start()

        timeout_seconds = int(os.getenv('ANALYZE_TIMEOUT_SECS', '300'))
        worker.join(timeout_seconds)

        if worker.is_alive():
            return jsonify({'error': f'Analysis timed out after {timeout_seconds}s. Try a shorter period/interval or confirm qwen model is loaded.'}), 504

        ok, payload = result_queue.get()
        if not ok:
            raise payload
        final_state = payload
        
        # Convert the result to a dictionary
        result = {
            'ticker': final_state.ticker,
            'period': final_state.period,
            'interval': final_state.interval,
            'horizon_days': final_state.horizon_days,
            'market': {
                'ticker': final_state.market.ticker if final_state.market else None,
                'period': final_state.market.period if final_state.market else None,
                'interval': final_state.market.interval if final_state.market else None,
                'price_csv': final_state.market.price_csv if final_state.market else None
            } if final_state.market else None,
            'news': {
                'ticker': final_state.news.ticker if final_state.news else None,
                'window_days': final_state.news.window_days if final_state.news else None,
                'items': [{'title': item.title, 'url': item.url, 'published': item.published} for item in final_state.news.items] if final_state.news and final_state.news.items else []
            } if final_state.news else None,
            'sentiment': {
                'ticker': final_state.sentiment.ticker if final_state.sentiment else None,
                'news_items_analyzed': final_state.sentiment.news_items_analyzed if final_state.sentiment else 0,
                'overall_sentiment': final_state.sentiment.overall_sentiment if final_state.sentiment else 'neutral',
                'confidence_score': final_state.sentiment.confidence_score if final_state.sentiment else 0.0,
                'summary': final_state.sentiment.summary if final_state.sentiment else '',
                'investment_recommendation': final_state.sentiment.investment_recommendation if final_state.sentiment else '',
                'key_insights': final_state.sentiment.key_insights if final_state.sentiment else [],
                'methodology': final_state.sentiment.methodology if final_state.sentiment else ''
            } if final_state.sentiment else None,
            'valuation': {
                'ticker': final_state.valuation.ticker if final_state.valuation else None,
                'analysis_period': final_state.valuation.analysis_period if final_state.valuation else None,
                'trading_days': final_state.valuation.trading_days if final_state.valuation else 0,
                'cumulative_return': final_state.valuation.cumulative_return if final_state.valuation else 0.0,
                'annualized_return': final_state.valuation.annualized_return if final_state.valuation else 0.0,
                'daily_volatility': final_state.valuation.daily_volatility if final_state.valuation else 0.0,
                'annualized_volatility': final_state.valuation.annualized_volatility if final_state.valuation else 0.0,
                'price_trend': final_state.valuation.price_trend if final_state.valuation else 'neutral',
                'volatility_regime': final_state.valuation.volatility_regime if final_state.valuation else 'medium',
                'valuation_insights': final_state.valuation.valuation_insights if final_state.valuation else [],
                'trend_analysis': final_state.valuation.trend_analysis if final_state.valuation else '',
                'risk_assessment': final_state.valuation.risk_assessment if final_state.valuation else '',
                'methodology': final_state.valuation.methodology if final_state.valuation else ''
            } if final_state.valuation else None,
            'metrics': {
                'ticker': final_state.metrics.ticker if final_state.metrics else None,
                'horizon_days': final_state.metrics.horizon_days if final_state.metrics else 0,
                'annual_vol': final_state.metrics.annual_vol if final_state.metrics else 0.0,
                'max_drawdown': final_state.metrics.max_drawdown if final_state.metrics else 0.0,
                'daily_var_95': final_state.metrics.daily_var_95 if final_state.metrics else 0.0,
                'sharpe_like': final_state.metrics.sharpe_like if final_state.metrics else None,
                'notes': final_state.metrics.notes if final_state.metrics else [],
                'risk_flags': final_state.metrics.risk_flags if final_state.metrics else []
            } if final_state.metrics else None,
            'report': {
                'ticker': final_state.report.ticker if final_state.report else None,
                'as_of': final_state.report.as_of if final_state.report else '',
                'summary': final_state.report.summary if final_state.report else '',
                'key_findings': final_state.report.key_findings if final_state.report else [],
                'metrics_table': final_state.report.metrics_table if final_state.report else {},
                'risk_flags': final_state.report.risk_flags if final_state.report else [],
                'methodology': final_state.report.methodology if final_state.report else '',
                'markdown_report': final_state.report.markdown_report if final_state.report else ''
            } if final_state.report else None
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze_stock: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Finance Risk Analysis API is running'})

@app.route('/api/models', methods=['GET'])
def get_available_models():
    return jsonify({
        'models': ['qwen3:4b'],
        'current_model': 'qwen3:4b',
        'description': 'Local Ollama Qwen 4B model'
    })

if __name__ == '__main__':
    print("🚀 Starting Finance Risk Analysis API...")
    print("📊 Multi-Agent System Ready")
    print("🤖 Using Local Ollama Qwen:4b Model")
    app.run(debug=True, host='0.0.0.0', port=5001)
