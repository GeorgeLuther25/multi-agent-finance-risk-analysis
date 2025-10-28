from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
import threading
import queue
import requests

# Ensure we can import project code from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import main  # src/main.py
from agents import State  # src/agents.py

app = Flask(__name__)
CORS(app)


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Finance Risk Analysis API is running'})


@app.route('/api/models', methods=['GET'])
def get_available_models():
    # Keep simple and static for now
    return jsonify({
        'models': ['qwen:4b'],
        'current_model': 'qwen:4b',
        'description': 'Local Ollama Qwen 4B model'
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.get_json() or {}

        # Validate required fields
        required = ['ticker', 'period', 'interval', 'horizon_days']
        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({'error': f"Missing required field(s): {', '.join(missing)}"}), 400

        state = State(
            ticker=str(data['ticker']).upper(),
            period=str(data['period']),
            interval=str(data['interval']),
            horizon_days=int(data['horizon_days']),
        )

        # Quick Ollama reachability check (fail fast)
        try:
            requests.get(os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434') + '/api/tags', timeout=2)
        except Exception:
            return jsonify({'error': 'Ollama is not reachable. Start it with: ollama serve'}), 503

        # Build graph
        graph = main.build_graph()

        # Run with watchdog timeout to avoid UI hang
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)

        def _run():
            try:
                res = graph.invoke(state)
                result_queue.put((True, res))
            except Exception as e:
                result_queue.put((False, e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        timeout_seconds = int(os.getenv('ANALYZE_TIMEOUT_SECS', '90'))
        thread.join(timeout_seconds)

        if thread.is_alive():
            return jsonify({'error': f'Analysis timed out after {timeout_seconds}s. Try a shorter period/interval or confirm qwen model is loaded.'}), 504

        ok, payload = result_queue.get()
        if not ok:
            raise payload

        final_state = payload

        # Serialize response
        result = {
            'ticker': final_state.ticker,
            'period': final_state.period,
            'interval': final_state.interval,
            'horizon_days': final_state.horizon_days,
            'market': (
                {
                    'ticker': final_state.market.ticker,
                    'period': final_state.market.period,
                    'interval': final_state.market.interval,
                    'price_csv': final_state.market.price_csv,
                }
                if getattr(final_state, 'market', None) else None
            ),
            'news': (
                {
                    'ticker': final_state.news.ticker,
                    'window_days': final_state.news.window_days,
                    'items': [
                        {'title': it.title, 'url': it.url, 'published': it.published}
                        for it in (final_state.news.items or [])
                    ],
                }
                if getattr(final_state, 'news', None) else None
            ),
            'sentiment': (
                {
                    'ticker': final_state.sentiment.ticker,
                    'news_items_analyzed': final_state.sentiment.news_items_analyzed,
                    'overall_sentiment': final_state.sentiment.overall_sentiment,
                    'confidence_score': final_state.sentiment.confidence_score,
                    'summary': final_state.sentiment.summary,
                    'investment_recommendation': final_state.sentiment.investment_recommendation,
                    'key_insights': final_state.sentiment.key_insights,
                    'methodology': final_state.sentiment.methodology,
                }
                if getattr(final_state, 'sentiment', None) else None
            ),
            'valuation': (
                {
                    'ticker': final_state.valuation.ticker,
                    'analysis_period': final_state.valuation.analysis_period,
                    'trading_days': final_state.valuation.trading_days,
                    'cumulative_return': final_state.valuation.cumulative_return,
                    'annualized_return': final_state.valuation.annualized_return,
                    'daily_volatility': final_state.valuation.daily_volatility,
                    'annualized_volatility': final_state.valuation.annualized_volatility,
                    'price_trend': final_state.valuation.price_trend,
                    'volatility_regime': final_state.valuation.volatility_regime,
                    'valuation_insights': final_state.valuation.valuation_insights,
                    'trend_analysis': final_state.valuation.trend_analysis,
                    'risk_assessment': final_state.valuation.risk_assessment,
                    'methodology': final_state.valuation.methodology,
                }
                if getattr(final_state, 'valuation', None) else None
            ),
            'metrics': (
                {
                    'ticker': final_state.metrics.ticker,
                    'horizon_days': final_state.metrics.horizon_days,
                    'annual_vol': final_state.metrics.annual_vol,
                    'max_drawdown': final_state.metrics.max_drawdown,
                    'daily_var_95': final_state.metrics.daily_var_95,
                    'sharpe_like': final_state.metrics.sharpe_like,
                    'notes': final_state.metrics.notes,
                    'risk_flags': final_state.metrics.risk_flags,
                }
                if getattr(final_state, 'metrics', None) else None
            ),
            'report': (
                {
                    'ticker': final_state.report.ticker,
                    'as_of': final_state.report.as_of,
                    'summary': final_state.report.summary,
                    'key_findings': final_state.report.key_findings,
                    'metrics_table': final_state.report.metrics_table,
                    'risk_flags': final_state.report.risk_flags,
                    'methodology': final_state.report.methodology,
                    'markdown_report': final_state.report.markdown_report,
                }
                if getattr(final_state, 'report', None) else None
            ),
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in analyze_stock: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("🚀 Starting Finance Risk Analysis API...")
    print("📊 Multi-Agent System Ready")
    print("🤖 Using Local Ollama Qwen:4b Model")
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', '5001')))


