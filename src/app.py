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
        
        # Handle both dict and State object responses
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        # Convert the result to a dictionary
        result = {
            'ticker': safe_get(final_state, 'ticker'),
            'period': safe_get(final_state, 'period'),
            'interval': safe_get(final_state, 'interval'),
            'horizon_days': safe_get(final_state, 'horizon_days'),
            'market': {
                'ticker': safe_get(safe_get(final_state, 'market'), 'ticker'),
                'period': safe_get(safe_get(final_state, 'market'), 'period'),
                'interval': safe_get(safe_get(final_state, 'market'), 'interval'),
                'price_csv': safe_get(safe_get(final_state, 'market'), 'price_csv')
            } if safe_get(final_state, 'market') else None,
            'news': {
                'ticker': safe_get(safe_get(final_state, 'news'), 'ticker'),
                'window_days': safe_get(safe_get(final_state, 'news'), 'window_days'),
                'items': [{'title': safe_get(item, 'title'), 'url': safe_get(item, 'url'), 'published': safe_get(item, 'published')} for item in (safe_get(safe_get(final_state, 'news'), 'items') or [])]
            } if safe_get(final_state, 'news') else None,
            'sentiment': {
                'ticker': safe_get(safe_get(final_state, 'sentiment'), 'ticker'),
                'news_items_analyzed': safe_get(safe_get(final_state, 'sentiment'), 'news_items_analyzed'),
                'overall_sentiment': safe_get(safe_get(final_state, 'sentiment'), 'overall_sentiment'),
                'confidence_score': safe_get(safe_get(final_state, 'sentiment'), 'confidence_score'),
                'summary': safe_get(safe_get(final_state, 'sentiment'), 'summary'),
                'investment_recommendation': safe_get(safe_get(final_state, 'sentiment'), 'investment_recommendation'),
                'key_insights': safe_get(safe_get(final_state, 'sentiment'), 'key_insights'),
                'methodology': safe_get(safe_get(final_state, 'sentiment'), 'methodology')
            } if safe_get(final_state, 'sentiment') else None,
            'valuation': {
                'ticker': safe_get(safe_get(final_state, 'valuation'), 'ticker'),
                'analysis_period': safe_get(safe_get(final_state, 'valuation'), 'analysis_period'),
                'trading_days': safe_get(safe_get(final_state, 'valuation'), 'trading_days'),
                'cumulative_return': safe_get(safe_get(final_state, 'valuation'), 'cumulative_return'),
                'annualized_return': safe_get(safe_get(final_state, 'valuation'), 'annualized_return'),
                'daily_volatility': safe_get(safe_get(final_state, 'valuation'), 'daily_volatility'),
                'annualized_volatility': safe_get(safe_get(final_state, 'valuation'), 'annualized_volatility'),
                'price_trend': safe_get(safe_get(final_state, 'valuation'), 'price_trend'),
                'volatility_regime': safe_get(safe_get(final_state, 'valuation'), 'volatility_regime'),
                'valuation_insights': safe_get(safe_get(final_state, 'valuation'), 'valuation_insights'),
                'trend_analysis': safe_get(safe_get(final_state, 'valuation'), 'trend_analysis'),
                'risk_assessment': safe_get(safe_get(final_state, 'valuation'), 'risk_assessment'),
                'methodology': safe_get(safe_get(final_state, 'valuation'), 'methodology')
            } if safe_get(final_state, 'valuation') else None,
            'metrics': {
                'ticker': safe_get(safe_get(final_state, 'metrics'), 'ticker'),
                'horizon_days': safe_get(safe_get(final_state, 'metrics'), 'horizon_days'),
                'annual_vol': safe_get(safe_get(final_state, 'metrics'), 'annual_vol'),
                'max_drawdown': safe_get(safe_get(final_state, 'metrics'), 'max_drawdown'),
                'daily_var_95': safe_get(safe_get(final_state, 'metrics'), 'daily_var_95'),
                'sharpe_like': safe_get(safe_get(final_state, 'metrics'), 'sharpe_like'),
                'notes': safe_get(safe_get(final_state, 'metrics'), 'notes'),
                'risk_flags': safe_get(safe_get(final_state, 'metrics'), 'risk_flags')
            } if safe_get(final_state, 'metrics') else None,
            'report': {
                'ticker': safe_get(safe_get(final_state, 'report'), 'ticker'),
                'as_of': safe_get(safe_get(final_state, 'report'), 'as_of'),
                'summary': safe_get(safe_get(final_state, 'report'), 'summary'),
                'key_findings': safe_get(safe_get(final_state, 'report'), 'key_findings'),
                'metrics_table': safe_get(safe_get(final_state, 'report'), 'metrics_table'),
                'risk_flags': safe_get(safe_get(final_state, 'report'), 'risk_flags'),
                'methodology': safe_get(safe_get(final_state, 'report'), 'methodology'),
                'markdown_report': safe_get(safe_get(final_state, 'report'), 'markdown_report')
            } if safe_get(final_state, 'report') else None
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
