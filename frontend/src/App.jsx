import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    ticker: 'AAPL',
    period: '1wk',
    interval: '1d',
    horizon_days: 30
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post('/api/analyze', formData);
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred while analyzing the stock');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1>🤖 Multi-Agent Finance Risk Analysis</h1>
          <p>Powered by LangGraph & Local Ollama Qwen Model</p>
        </div>

        {/* Input Form */}
        <div className="form-container">
          <h2>📊 Analysis Parameters</h2>
          
          <form onSubmit={handleSubmit} className="form">
            <div className="form-group">
              <label>Stock Ticker</label>
              <input
                type="text"
                name="ticker"
                value={formData.ticker}
                onChange={handleInputChange}
                placeholder="e.g., AAPL, MSFT, GOOGL"
                required
              />
            </div>

            <div className="form-group">
              <label>Time Period</label>
              <select
                name="period"
                value={formData.period}
                onChange={handleInputChange}
              >
                <option value="1d">1 Day</option>
                <option value="5d">5 Days</option>
                <option value="1mo">1 Month</option>
                <option value="3mo">3 Months</option>
                <option value="6mo">6 Months</option>
                <option value="1y">1 Year</option>
                <option value="2y">2 Years</option>
              </select>
            </div>

            <div className="form-group">
              <label>Interval</label>
              <select
                name="interval"
                value={formData.interval}
                onChange={handleInputChange}
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="30m">30 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="1d">1 Day</option>
                <option value="1wk">1 Week</option>
              </select>
            </div>

            <div className="form-group">
              <label>Risk Horizon (Days)</label>
              <input
                type="number"
                name="horizon_days"
                value={formData.horizon_days}
                onChange={handleInputChange}
                min="1"
                max="365"
                required
              />
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? 'Analyzing...' : '📈 Analyze Stock Risk'}
            </button>
          </form>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="results">
            {/* Summary Cards */}
            <div className="summary-cards">
              <div className="card">
                <h3>Cumulative Return</h3>
                <p className={result.valuation?.cumulative_return >= 0 ? 'positive' : 'negative'}>
                  {(result.valuation?.cumulative_return * 100).toFixed(2)}%
                </p>
              </div>

              <div className="card">
                <h3>Annualized Volatility</h3>
                <p className="volatility">
                  {(result.valuation?.annualized_volatility * 100).toFixed(1)}%
                </p>
              </div>

              <div className="card">
                <h3>Price Trend</h3>
                <p className="trend">{result.valuation?.price_trend}</p>
              </div>

              <div className="card">
                <h3>Risk Flags</h3>
                <p className="risk-flags">{result.metrics?.risk_flags?.length || 0}</p>
              </div>
            </div>

            {/* Detailed Report */}
            <div className="report">
              <h2>📋 Detailed Risk Analysis Report</h2>
              <div className="markdown-content">
                <ReactMarkdown>{result.report?.markdown_report}</ReactMarkdown>
              </div>
            </div>

            {/* Raw Data */}
            <details className="raw-data">
              <summary>Raw Analysis Data</summary>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;