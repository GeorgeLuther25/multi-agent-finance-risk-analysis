
# 🤖 Multi-Agent Finance Risk Analysis

Advanced multi-agent system for comprehensive financial analysis using LangChain, RAG (Retrieval-Augmented Generation), and real-time data. Analyzes stocks through sentiment analysis, fundamental analysis using 10-K/10-Q filings, technical analysis, and risk assessment.

## 📁 Project Structure

```
multi-agent-finance-risk-analysis/
├── README.md                        # This documentation
├── requirements.txt                 # Python dependencies
├── .env.example                    # Environment variables template
├── src/
│   ├── __init__.py
│   ├── main.py                     # Main application entry point
│   ├── config.py                   # LLM and API configuration
│   ├── schemas.py                  # Pydantic data models
│   ├── agents.py                   # Multi-agent definitions
│   │   ├── data_agent()           # Market data and news retrieval
│   │   ├── fundamental_agent()    # RAG-powered 10-K/10-Q analysis
│   │   ├── sentiment_agent()      # News sentiment analysis
│   │   ├── valuation_agent()      # Technical analysis metrics
│   │   ├── risk_agent()           # Risk assessment calculations
│   │   └── writer_agent()         # Final report generation
│   ├── tools.py                    # LangChain tools and utilities
│   ├── rag_utils.py               # RAG system for fundamental analysis
│   │   ├── FundamentalRAG         # Main RAG class
│   │   ├── ingest_document()      # Single document ingestion
│   │   ├── retrieve_relevant_chunks() # Document querying
│   │   └── initialize_sample_data() # Sample data setup
│   └── __pycache__/               # Python bytecode cache
├── data/                          # Data storage directory
│   ├── chroma_db/                # ChromaDB vector database (auto-created)
│   ├── sample_filings/           # Sample text files (auto-created)
│   └── filings/                  # Your PDF documents (create manually)
└── .venv/                        # Virtual environment (auto-created)
```

## 🚀 Agent Workflow

The system processes analysis through these specialized agents:

1. **Data Agent** → Fetches market data and recent news
2. **Fundamental Agent** → Analyzes 10-K/10-Q filings using RAG
3. **Sentiment Agent** → Analyzes news sentiment with reflection
4. **Valuation Agent** → Computes technical analysis metrics
5. **Risk Agent** → Calculates risk metrics and flags
6. **Writer Agent** → Generates comprehensive markdown report

## 📚 Additional Documentation

- **LangSmith Traces**: Monitor agent execution at [smith.langchain.com](https://smith.langchain.com/)

## 🔧 Troubleshooting

### Common Issues

#### Fundamental Agent Issues
- **"OpenAI API key not set"**: Ensure `OPENAI_API_KEY` is set in your environment
- **"ChromaDB errors"**: Delete `./data/chroma_db/` and reinitialize with sample data
- **"Agent timeout"**: Increase `max_iterations` in the fundamental agent configuration

#### General Issues
- **Import errors**: Ensure virtual environment is activated and dependencies are installed
- **No news data**: Check `POLYGON_API_KEY` is set and valid
- **LangSmith not working**: Verify `LANGCHAIN_API_KEY` and `LANGCHAIN_TRACING_V2=true`

### Environment Variables Checklist
```bash
# Required for basic functionality (choose one)
export OPENAI_API_KEY="your-openai-key"    # For OpenAI models
# OR use local Ollama (no API key needed)
export MODEL_PROVIDER="ollama"             # Force Ollama usage
export OLLAMA_MODEL="llama3.2:3b"         # Optional: specify model

# Required for market data
export POLYGON_API_KEY="your-polygon-key"

# Optional for LangSmith tracing
export LANGCHAIN_API_KEY="ls_your-langsmith-key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="Multi-Agent Finance Bot"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test with sample data
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details., technical analysis, and risk assessment.

## 🌟 Key Features

- **Multi-Agent Architecture**: Specialized agents for different analysis types
- **RAG-Powered Fundamental Analysis**: Analyzes 10-K/10-Q SEC filings using vector search
- **Real-Time Data Integration**: Live news, market data, and sentiment analysis
- **Risk Assessment**: Traditional financial risk metrics with automated flags
- **LangSmith Integration**: Complete workflow tracing and debugging
- **Document Ingestion**: Custom tools for ingesting financial documents

## 🤖 Agent Overview

The system employs 6 specialized agents that work together:

| Agent | Purpose | Key Features |
|-------|---------|--------------|
| **Data Agent** | Market data & news retrieval | Real-time price data, news extraction, content enhancement |
| **Fundamental Agent** | 10-K/10-Q analysis | RAG-powered SEC filing analysis, financial metrics, business insights |
| **Sentiment Agent** | News sentiment analysis | Reflection-enhanced prompting, confidence scoring, investment recommendations |
| **Valuation Agent** | Technical analysis | Volatility calculations, trend analysis, price metrics |
| **Risk Agent** | Risk assessment | VaR calculations, drawdown analysis, risk flags |
| **Writer Agent** | Report generation | Comprehensive markdown reports, final analysis compilation |

### Fundamental Agent Highlights

The **Fundamental Agent** is powered by a sophisticated RAG (Retrieval-Augmented Generation) system:

- 🔍 **Semantic Search**: ChromaDB vector database for intelligent document retrieval
- 📊 **Comprehensive Analysis**: Automated extraction of financial metrics, risk factors, and business insights
- 🛠️ **Efficient Querying**: Batched queries to minimize API calls and maximize information retrieval
- 📈 **Health Scoring**: Automated financial health scoring (0-10 scale)
- 🎯 **Investment Thesis**: AI-generated investment recommendations based on SEC filings

## 📋 Prerequisites

- Python 3.8+
- macOS/Linux/Windows
- Internet connection (for stock data)

## ��️ Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd multi-agent-finance-risk-analysis
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 🔑 LangSmith Setup (Optional but Recommended)

### Step 1: Get LangSmith API Key

1. Go to [https://smith.langchain.com/](https://smith.langchain.com/)
2. Sign up/Login with your account
3. Navigate to **Settings** → **API Keys**
4. Create a new API key
5. Copy the key (starts with `ls_`)

### Step 2: Configure Environment Variables

#### Option A: Export in Terminal (Temporary)

```bash
export LANGCHAIN_API_KEY="ls_your_api_key_here"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="Multi-Agent Finance Bot"

export OPENAI_API_KEY=ls_your_api_key_here
# OR use local Ollama (no API key needed)
export MODEL_PROVIDER="ollama"             # Force Ollama usage
export OLLAMA_MODEL="llama3.2:3b"         # Optional: specify 
export POLYGON_API_KEY=ls_your_api_key_here
```

#### Option B: Create .env File (Recommended)

Create a `.env` file in your project root:

```bash
# .env
LANGCHAIN_API_KEY=ls_your_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT="Multi-Agent Finance Bot"

OPENAI_API_KEY=ls_your_api_key_here
# OR use local Ollama (no API key needed)
OLLAMA_MODEL="llama3.2:3b"

POLYGON_API_KEY=ls_your_api_key_here
```

Then load it before running:

```bash
source .env
```

<!-- #### Option C: Set in main.py (Code-based)

Edit `src/main.py` and ensure these lines are uncommented:

```python
# LangSmith configuration
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "Multi-Agent Finance Bot")
``` -->

## �� Running the System

### Basic Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the multi-agent system
python -m src.main
```

### With LangSmith Enabled

```bash
# Activate virtual environment
source .venv/bin/activate

# Set API key (if using export method)
export LANGCHAIN_API_KEY="ls_your_api_key_here"

# Run with tracing enabled
python -m src.main
```

### Custom Parameters

Edit `src/main.py` to change default parameters:

```python
# Change ticker, period, or horizon
state = State(
    ticker="MSFT",        # Change stock symbol
    period="6mo",         # Change time period (1y, 6mo, 2y)
    interval="1d",        # Change interval (1d, 1h, 1wk)
    horizon_days=60       # Change risk horizon
)
```

## 📊 What You'll See

### Console Output

The system will display:
- **News Retrieval**: Real-time news fetching with URL content extraction
- **Fundamental Analysis**: RAG-powered analysis of 10-K/10-Q SEC filings with comprehensive insights
- **Valuation Analysis**: Computational metrics (annualized returns, volatility using 252-day formula)
- **Sentiment Analysis**: Reflection-enhanced prompting with confidence scores
- **Risk Metrics**: Traditional financial risk calculations and flags
- **Debug Information**: Data processing steps and agent workflow
- **Final Report**: Complete Markdown risk analysis with all sections
- **Final State**: Complete state object with all agent outputs

### LangSmith Platform (if enabled)

Visit [https://smith.langchain.com/](https://smith.langchain.com/) to see:
- **Project Dashboard**: "Multi-Agent Finance Bot"
- **Execution Traces**: Complete workflow runs
- **Agent Performance**: Timing and success rates
- **Data Flow**: Input/output between agents
- **Debugging**: Detailed execution logs

## 🔧 Configuration Options

### LLM Configuration

Edit `src/config.py` to change the language model:

```python
# Use OpenAI (requires API key)
MODEL_PROVIDER="openai" python -m src.main

# Use local Ollama (free, runs locally)
MODEL_PROVIDER="ollama" python -m src.main

# Auto-detect (default)
python -m src.main
```

#### Setting up Ollama (Free Local Models)

1. **Install Ollama**:
   ```bash
   # macOS
   brew install ollama
   
   # Linux
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Windows: Download from https://ollama.com/
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Pull a model** (choose based on your system):
   ```bash
    # Lightweight models (good for testing)
    ollama pull llama3.2:1b      # 1.3GB")
    ollama pull llama3.2:3b      # 2.0GB")
    
    # More capable models
    ollama pull llama3.1:8b      # 4.7GB")
    ollama pull llama3.1:70b     # 40GB (requires 64GB+ RAM)")
    ollama pull qwen2.5:7b       # 4.4GB")
    
    # Specialized models
    ollama pull codellama:7b     # 3.8GB (for code)")
    ollama pull mistral:7b       # 4.1GB (general purpose)")
   ```

4. **Use with analysis**:
   ```bash
   export OLLAMA_MODEL=llama3.2:3b  # Optional: specify model
   MODEL_PROVIDER=ollama python -m src.main
   ```

### Risk Thresholds

Edit `src/agents.py` to adjust risk flags:

```python
# Customize risk thresholds
if stats["annual_vol"] > 0.45: flags.append("HIGH_VOLATILITY")
if stats["max_drawdown"] < -0.25: flags.append("DEEP_DRAWDOWN")
```

## � Fundamental Analysis & RAG System

The system includes a sophisticated Fundamental Analysis Agent that uses RAG (Retrieval-Augmented Generation) to analyze 10-K and 10-Q SEC filings.

### Features

- **Document Ingestion**: Custom tools to ingest PDF financial documents
- **Vector Search**: ChromaDB-powered semantic search across financial filings
- **Agent-Based Analysis**: LangChain agent with RAG tools for comprehensive analysis
- **Batch Queries**: Efficient single-call analysis with multiple queries
- **Sample Data**: Pre-loaded sample data for AAPL, MSFT, and GOOGL

### Quick Start with Sample Data

```bash
# Run analysis (will create a chroma DB with data from ./data/filings folder)
python -m src.main
```

### What the Fundamental Agent Analyzes

The agent performs comprehensive analysis including:
- **Executive Summary**: High-level company overview
- **Key Financial Metrics**: Revenue, margins, profitability metrics
- **Business Highlights**: Core business segments and products
- **Risk Assessment**: Identified risk factors from filings
- **Competitive Position**: Market positioning analysis
- **Growth Prospects**: Future outlook and opportunities
- **Financial Health Score**: Numerical rating (0-10)
- **Investment Thesis**: Comprehensive investment recommendation

### RAG System Architecture

1. **Document Ingestion**: PDF documents are loaded and chunked
2. **Vector Storage**: ChromaDB stores embeddings for semantic search
3. **Query Processing**: Natural language queries are converted to vector searches
4. **Context Retrieval**: Relevant document chunks are retrieved
5. **LLM Analysis**: OpenAI models analyze context to provide insights

### Advanced Usage

#### Custom Queries
The system supports sophisticated queries like:
- "What are the main revenue streams and their growth rates?"
- "Analyze the risk factors and their potential impact"
- "Compare business segments and their profitability"
- "What regulatory challenges does the company face?"

#### Programmatic Access
```python
from src.rag_utils import FundamentalRAG

# Initialize RAG system
rag = FundamentalRAG()

# Query documents
results = rag.retrieve_relevant_chunks("AAPL", "revenue growth analysis")

# Process results
for chunk in results:
    print(chunk.page_content)
```

## �📁 Project Structure
