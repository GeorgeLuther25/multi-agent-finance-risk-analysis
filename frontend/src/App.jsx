import React, { useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import jsPDF from 'jspdf';
import './App.css';

const initialFormState = {
    ticker: 'AAPL',
    period: '1wk',
    interval: '1d',
    horizon_days: 30,
    mode: 'chain' // change this to chain or debate mode
};

function App() {
  const [formData, setFormData] = useState(initialFormState);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [promptStatus, setPromptStatus] = useState(null);
  const [downloading, setDownloading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const parsePrompt = () => {
    if (!prompt.trim()) {
      setPromptStatus({ type: 'error', message: 'Enter a natural language request to parse.' });
      return;
    }

    const normalized = prompt.trim();
    const updates = {};

    const stopwords = new Set([
      'give', 'me', 'for', 'the', 'last', 'past', 'stock', 'stocks', 'shares', 'please',
      'show', 'get', 'pull', 'interval', 'intervals', 'of', 'at', 'a', 'an', 'analysis',
      'report', 'risk', 'horizon', 'days', 'day', 'over', 'next', 'with', 'and', 'to',
      'year', 'years', 'month', 'months', 'week', 'weeks', 'data'
    ]);

    const tickerMatch =
      normalized.match(/\bstock\s+(?:for|of)\s+([A-Za-z]{1,5})\b/i) ||
      normalized.match(/\b([A-Za-z]{1,5})\b\s*(?:stock|shares|equity)/i);

    if (tickerMatch) {
      updates.ticker = tickerMatch[1].toUpperCase();
    } else {
      const candidateTickers = normalized.match(/\b[A-Za-z]{1,5}\b/g) || [];
      const tickerCandidate = candidateTickers.find(
        token => !stopwords.has(token.toLowerCase())
      );
      if (tickerCandidate) {
        updates.ticker = tickerCandidate.toUpperCase();
      }
    }

    const periodMatch = normalized.match(/(?:last|past)\s+(\d+)\s*(day|days|week|weeks|month|months|year|years)/i);
    if (periodMatch) {
      const count = parseInt(periodMatch[1], 10);
      const unit = periodMatch[2].toLowerCase();
      if (!Number.isNaN(count) && count > 0) {
        if (unit.startsWith('day')) {
          updates.period = count <= 1 ? '1d' : count <= 5 ? '5d' : '1mo';
        } else if (unit.startsWith('week')) {
          updates.period = count <= 1 ? '1wk' : '1mo';
        } else if (unit.startsWith('month')) {
          if (count === 1) updates.period = '1mo';
          else if (count === 3) updates.period = '3mo';
          else if (count === 6) updates.period = '6mo';
          else if (count >= 12) updates.period = '1y';
        } else if (unit.startsWith('year')) {
          updates.period = count >= 2 ? '2y' : '1y';
        }
      }
    }

    const intervalMatch = normalized.match(/intervals?\s*(?:of|at)?\s*(?:every\s*)?(\d+)?\s*(minute|minutes|min|hour|hours|day|days|daily|week|weekly)/i);
    if (intervalMatch) {
      const count = intervalMatch[1] ? parseInt(intervalMatch[1], 10) : 1;
      const unit = intervalMatch[2].toLowerCase();
      if (unit.startsWith('min')) {
        if (count === 1) updates.interval = '1m';
        else if (count === 5) updates.interval = '5m';
        else if (count === 15) updates.interval = '15m';
        else if (count === 30) updates.interval = '30m';
      } else if (unit.startsWith('hour')) {
        updates.interval = '1h';
      } else if (unit.startsWith('day') || unit === 'daily') {
        updates.interval = '1d';
      } else if (unit.startsWith('week') || unit === 'weekly') {
        updates.interval = '1wk';
      }
    }

    const horizonMatch =
      normalized.match(/(?:risk\s+horizon|horizon|next)\s*(?:of|for|around)?\s*(\d+)\s*day/i) ||
      normalized.match(/(?:over|for)\s+the\s+next\s+(\d+)\s*day/i);
    if (horizonMatch) {
      const days = parseInt(horizonMatch[1], 10);
      if (!Number.isNaN(days) && days > 0) {
        updates.horizon_days = Math.min(Math.max(days, 1), 365);
      }
    }

    if (Object.keys(updates).length === 0) {
      setPromptStatus({
        type: 'error',
        message: 'Could not parse the request. Update the form manually or try a different phrasing.'
      });
      return;
    }

    setFormData(prev => ({
      ...prev,
      ...updates
    }));

    const summary = Object.entries(updates)
      .map(([key, value]) => `${key.replace('_', ' ')} → ${value}`)
      .join(', ');

    setPromptStatus({
      type: 'success',
      message: `Applied prompt: ${summary}`
    });
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

  const handleDownloadPdf = () => {
    if (!result?.report?.markdown_report || downloading) {
      return;
    }

    setDownloading(true);
    try {
      const doc = new jsPDF({
        orientation: 'portrait',
        unit: 'pt',
        format: 'a4',
      });

      const margin = 40;
      const pageWidth = doc.internal.pageSize.getWidth();
      const usableWidth = pageWidth - margin * 2;
      const lineHeight = 16;
      let cursorY = margin;

      const title = `Finance Risk Analysis — ${result.ticker || ''}`.trim();
      doc.setFont('Helvetica', 'bold');
      doc.setFontSize(18);
      doc.text(title, margin, cursorY);
      cursorY += lineHeight * 2;

      const sections = result.report.markdown_report
        .replace(/#+\s?/g, '')
        .replace(/\*\*/g, '')
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean);

      doc.setFont('Helvetica', 'normal');
      doc.setFontSize(11);

      sections.forEach(line => {
        const wrapped = doc.splitTextToSize(line, usableWidth);
        wrapped.forEach(textLine => {
          if (cursorY + lineHeight > doc.internal.pageSize.getHeight() - margin) {
            doc.addPage();
            cursorY = margin;
          }
          doc.text(textLine, margin, cursorY);
          cursorY += lineHeight;
        });
        cursorY += lineHeight * 0.5;
      });

      const filename = `${result.ticker || 'analysis'}_risk_report.pdf`;
      doc.save(filename);
    } catch (err) {
      console.error('Failed to export PDF:', err);
      setPromptStatus({
        type: 'error',
        message: 'Unable to download PDF. Please try again.',
      });
    } finally {
      setDownloading(false);
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

          <div className="prompt-helper">
            <label htmlFor="prompt">Describe your request</label>
            <textarea
              id="prompt"
              placeholder='e.g., "Give me AAPL stock for the last 2 years at intervals of 1 day with a 45 day risk horizon."'
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <button type="button" className="prompt-btn" onClick={parsePrompt}>
              🎯 Apply Prompt
            </button>
            {promptStatus && (
              <div className={`prompt-status ${promptStatus.type}`}>
                {promptStatus.message}
              </div>
            )}
          </div>
          
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

            <div className="form-group">
              <label>Agent Mode</label>
              <select
                name="mode"
                value={formData.mode}
                onChange={handleInputChange}
              >
                <option value="chain">Standard Pipeline</option>
                <option value="debate">Leader Debate</option>
              </select>
            </div>

            <button type="submit" disabled={loading} className="submit-btn">
              {loading ? 'Analyzing...' : '📈 Analyze Stock Risk'}
            </button>
            {result?.report?.markdown_report && (
              <button
                type="button"
                className="download-btn"
                onClick={handleDownloadPdf}
                disabled={downloading}
              >
                {downloading ? 'Preparing PDF…' : '📄 Download Report PDF'}
              </button>
            )}
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

            <div className="graph-card">
              <h3>Agent Orchestration</h3>
              <p className="graph-caption">
                {result.analysis_mode === 'debate'
                  ? 'Leader debates with every specialist twice before final synthesis.'
                  : 'Linear LangGraph chain executes agents in sequence.'}
              </p>
              <img
                src={`/visualizations/langgraph_${result.analysis_mode || 'chain'}.svg`}
                alt={`LangGraph ${result.analysis_mode || 'chain'} workflow`}
              />
            </div>

            {/* Detailed Report */}
            <div className="report">
              <h2>📋 Detailed Risk Analysis Report</h2>
              <div className="markdown-content">
                <ReactMarkdown>{result.report?.markdown_report}</ReactMarkdown>
              </div>
            </div>

            {result.debate_transcript && (
              <details className="debate-log">
                <summary>Debate Transcript</summary>
                <ul>
                  {result.debate_transcript.map((line, idx) => (
                    <li key={idx}>{line}</li>
                  ))}
                </ul>
              </details>
            )}

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
