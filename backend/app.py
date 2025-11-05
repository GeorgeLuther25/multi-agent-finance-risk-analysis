from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import traceback
import threading
import queue
import requests
import io
import base64
import openai

# Ensure we can import project code from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src import main  # src/main.py
from agents import State  # src/agents.py
from langchain_core.runnables import RunnableConfig
from utils.schemas import DebateReport

from markdown_pdf import MarkdownPdf, Section

app = Flask(__name__)
CORS(app)


def _detect_model_provider():
    raw = (os.getenv('MODEL_PROVIDER') or 'auto').strip().lower()
    if not raw:
        raw = 'auto'
    openai_key = os.getenv('OPENAI_API_KEY')
    if raw == 'auto':
        provider = 'openai' if openai_key else 'ollama'
    else:
        provider = raw
    return provider, openai_key


def _is_ollama_available():
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').rstrip('/')
    try:
        requests.get(f"{base_url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Finance Risk Analysis API is running'})


@app.route('/favicon.ico')
def favicon_placeholder():
    """Return an empty response so browsers stop logging 404s for favicons."""
    return ('', 204)


@app.route('/api/models', methods=['GET'])
def get_available_models():
    raw_provider = (os.getenv('MODEL_PROVIDER') or 'auto').strip().lower() or 'auto'
    provider, openai_key = _detect_model_provider()

    if provider == 'openai':
        if not openai_key:
            return jsonify({
                'error': (
                    "OpenAI GPT-4o not configured. Set OPENAI_API_KEY or set MODEL_PROVIDER=ollama."
                )
            }), 503
        current_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        description = 'OpenAI GPT model'
        models = [current_model]
    elif provider == 'ollama':
        if not _is_ollama_available():
            return jsonify({
                'error': 'Ollama is not reachable. Start it with: ollama serve'
            }), 503
        current_model = os.getenv('OLLAMA_MODEL', 'qwen3:4b')
        description = 'Local Ollama model'
        models = [current_model]
    else:
        return jsonify({
            'error': f"Unknown MODEL_PROVIDER: {raw_provider}. Must be 'openai' or 'ollama'."
        }), 400

    return jsonify({
        'provider': provider,
        'models': models,
        'current_model': current_model,
        'description': description
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

        requested_mode = str(data.get('mode', '')).strip().lower() or None
        env_mode = os.getenv('ANALYSIS_MODE')
        effective_mode = requested_mode or env_mode

        graph, state_cls = main.get_workflow(effective_mode)
        print(f"🧠 Analysis mode: {main.resolve_mode(effective_mode)}")

        state = state_cls(
            ticker=str(data['ticker']).upper(),
            period=str(data['period']),
            interval=str(data['interval']),
            horizon_days=int(data['horizon_days']),
        )

        # Detect provider dynamically (OpenAI vs Ollama)
        model_provider, openai_key = _detect_model_provider()

        if model_provider == 'openai':
            if not openai_key:
                return jsonify({
                    'error': (
                        "OpenAI GPT-4o not configured. Set OPENAI_API_KEY or set MODEL_PROVIDER=ollama."
                    )
                }), 503
            print("🔍 Using OpenAI provider (GPT-4o). Skipping Ollama check.")
        elif model_provider == 'ollama':
            if not _is_ollama_available():
                return jsonify({'error': 'Ollama is not reachable. Start it with: ollama serve'}), 503
        else:
            return jsonify({
                'error': f"Unknown MODEL_PROVIDER: {model_provider}. Must be 'openai' or 'ollama'."
            }), 400

        # Log which provider is being used (for debugging)
        print(f"🔍 Using provider: {model_provider}")

        # Run with watchdog timeout to avoid UI hang
        result_queue: "queue.Queue" = queue.Queue(maxsize=1)

        def _run():
            try:
                res = graph.invoke(state)
                res = State(**res)

                # # Debate model for final recommendation
                finalRecommendationGraph = main.build_final_recommendation_graph()
                debateReport = DebateReport(agent_list=["fundamental", "sentiment", "valuation"])
                debateReport.agent_max_turn = 5
                res.debate = debateReport
                res = finalRecommendationGraph.invoke(res, config=RunnableConfig(recursion_limit=100), verbose=True)
                res = State(**res)

                # Use the output directly, to test print report only
                # with open("final_state_with_debate.json", "r") as f:
                #     res = State.model_validate_json(f.read())
                #     print(res)
                
                pdf = MarkdownPdf(toc_level=2, optimize=True)
                pdf.add_section(Section(res.report.markdown_report))
                out = io.BytesIO()
                pdf.save_bytes(out)
                out.seek(0)

                # Put result in queue
                result_queue.put((True, res, out))
            except Exception as e:
                result_queue.put((False, e, None))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        timeout_seconds = int(os.getenv('ANALYZE_TIMEOUT_SECS', '900'))
        thread.join(timeout_seconds)

        if thread.is_alive():
            return jsonify({'error': f'Analysis timed out after {timeout_seconds}s. Try a shorter period/interval.'}), 504

        ok, payload, out_pdf = result_queue.get()
        if not ok:
            raise payload

        final_state = payload

        # ✅ Fix: normalize final_state to dict if needed
        if not isinstance(final_state, dict):
            final_state = final_state.__dict__ if hasattr(final_state, "__dict__") else {}

        # ✅ Safe getter to handle both dicts and objects
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # ✅ Serialize response safely (kept identical)
        result = {
            'ticker': safe_get(final_state, 'ticker'),
            'period': safe_get(final_state, 'period'),
            'interval': safe_get(final_state, 'interval'),
            'horizon_days': safe_get(final_state, 'horizon_days'),
            'market': (
                {
                    'ticker': safe_get(safe_get(final_state, 'market'), 'ticker'),
                    'period': safe_get(safe_get(final_state, 'market'), 'period'),
                    'interval': safe_get(safe_get(final_state, 'market'), 'interval'),
                    'price_csv': safe_get(safe_get(final_state, 'market'), 'price_csv'),
                }
                if safe_get(final_state, 'market') else None
            ),
            'news': (
                {
                    'ticker': safe_get(safe_get(final_state, 'news'), 'ticker'),
                    'window_days': safe_get(safe_get(final_state, 'news'), 'window_days'),
                    'items': [
                        {
                            'title': getattr(it, 'title', safe_get(it, 'title')),
                            'url': getattr(it, 'url', safe_get(it, 'url')),
                            'published': getattr(it, 'published', safe_get(it, 'published')),
                        }
                        for it in (safe_get(safe_get(final_state, 'news'), 'items') or [])
                    ],
                }
                if safe_get(final_state, 'news') else None
            ),
            'sentiment': (
                {
                    'ticker': safe_get(safe_get(final_state, 'sentiment'), 'ticker'),
                    'news_items_analyzed': safe_get(safe_get(final_state, 'sentiment'), 'news_items_analyzed'),
                    'overall_sentiment': safe_get(safe_get(final_state, 'sentiment'), 'overall_sentiment'),
                    'confidence_score': safe_get(safe_get(final_state, 'sentiment'), 'confidence_score'),
                    'summary': safe_get(safe_get(final_state, 'sentiment'), 'summary'),
                    'investment_recommendation': safe_get(safe_get(final_state, 'sentiment'), 'investment_recommendation'),
                    'key_insights': safe_get(safe_get(final_state, 'sentiment'), 'key_insights'),
                    'methodology': safe_get(safe_get(final_state, 'sentiment'), 'methodology'),
                }
                if safe_get(final_state, 'sentiment') else None
            ),
            'valuation': (
                {
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
                    'methodology': safe_get(safe_get(final_state, 'valuation'), 'methodology'),
                }
                if safe_get(final_state, 'valuation') else None
            ),
            'metrics': (
                {
                    'ticker': safe_get(safe_get(final_state, 'metrics'), 'ticker'),
                    'horizon_days': safe_get(safe_get(final_state, 'metrics'), 'horizon_days'),
                    'annual_vol': safe_get(safe_get(final_state, 'metrics'), 'annual_vol'),
                    'max_drawdown': safe_get(safe_get(final_state, 'metrics'), 'max_drawdown'),
                    'daily_var_95': safe_get(safe_get(final_state, 'metrics'), 'daily_var_95'),
                    'sharpe_like': safe_get(safe_get(final_state, 'metrics'), 'sharpe_like'),
                    'notes': safe_get(safe_get(final_state, 'metrics'), 'notes'),
                    'risk_flags': safe_get(safe_get(final_state, 'metrics'), 'risk_flags'),
                }
                if safe_get(final_state, 'metrics') else None
            ),
            'report': (
                {
                    'ticker': safe_get(safe_get(final_state, 'report'), 'ticker'),
                    'as_of': safe_get(safe_get(final_state, 'report'), 'as_of'),
                    'summary': safe_get(safe_get(final_state, 'report'), 'summary'),
                    'key_findings': safe_get(safe_get(final_state, 'report'), 'key_findings'),
                    'metrics_table': safe_get(safe_get(final_state, 'report'), 'metrics_table'),
                    'risk_flags': safe_get(safe_get(final_state, 'report'), 'risk_flags'),
                    'methodology': safe_get(safe_get(final_state, 'report'), 'methodology'),
                    'markdown_report': safe_get(safe_get(final_state, 'report'), 'markdown_report'),
                    'consensus_summary': safe_get(safe_get(safe_get(final_state, 'report'), 'debate'),'consensus_summary'),
                }
                if safe_get(final_state, 'report') else None
            ),
            'debate': (
                {
                    'consensus_summary': safe_get(safe_get(final_state, 'debate'),'consensus_summary'),
                }
                if safe_get(final_state, 'debate') else None
            )
        }

        result['analysis_mode'] = main.resolve_mode(effective_mode)

        transcript = safe_get(safe_get(final_state, 'debate'), 'agent_arguments')
        if transcript:
            result['agent_arguments'] = transcript

        if out_pdf:
            result['report_pdf_base64'] = base64.b64encode(out_pdf.read()).decode('utf-8')

        return jsonify(result)

    except openai.AuthenticationError:
        return jsonify({'error': 'OpenAI authentication failed. Set a valid OPENAI_API_KEY.'}), 502
    except openai.OpenAIError as e:
        print(f"OpenAI API error: {e}")
        return jsonify({'error': f'Analysis failed due to OpenAI API error: {str(e)}'}), 502
    except Exception as e:
        print(f"Error in analyze_stock: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("🚀 Starting Finance Risk Analysis API...")
    print("📊 Multi-Agent System Ready")
    provider, _openai_key = _detect_model_provider()
    if provider == 'openai':
        print(f"🤖 Using OpenAI model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}")
    else:
        print(f"🤖 Using Ollama model: {os.getenv('OLLAMA_MODEL', 'qwen3:4b')}")
    app.run(
        debug=True,
        host=os.getenv('HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', '5050')),
    )
