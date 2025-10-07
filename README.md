
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
# Use HuggingFace free tier
MODEL_NAME = "microsoft/DialoGPT-medium"

# Use local Ollama (if installed)
MODEL_NAME = "gemma:2b"
```

### Risk Thresholds

Edit `src/agents.py` to adjust risk flags:

```python
# Customize risk thresholds
if stats["annual_vol"] > 0.45: flags.append("HIGH_VOLATILITY")
if stats["max_drawdown"] < -0.25: flags.append("DEEP_DRAWDOWN")
```

## 📁 Project Structure
