# 🤖 Multi-Agent Finance Risk Analysis - React UI

A modern React frontend for the Multi-Agent Finance Risk Analysis system powered by LangGraph and local Ollama Qwen models.

## 🎯 Features

- **Modern React UI** with Tailwind CSS styling
- **Real-time Stock Analysis** with interactive forms
- **Multi-Agent System Integration** via Flask API
- **Local Ollama Support** using your Qwen:4b model
- **Comprehensive Risk Reports** with markdown rendering
- **Responsive Design** for desktop and mobile

## 📁 Project Structure

```
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── App.jsx         # Main React component
│   │   ├── App.css         # Tailwind CSS styles
│   │   ├── index.js        # React entry point
│   │   └── index.css       # Base styles
│   ├── public/
│   │   └── index.html      # HTML template
│   └── package.json        # Frontend dependencies
├── backend/                 # Flask API backend
│   ├── app.py             # Flask application
│   └── requirements.txt    # Backend dependencies
├── src/                    # Original Python multi-agent system
└── setup.sh               # Automated setup script
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the automated setup
./setup.sh
```

### Option 2: Manual Setup

#### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

#### 2. Install Backend Dependencies

```bash
cd backend
pip3 install -r requirements.txt
```

#### 3. Install Main Project Dependencies

```bash
cd ..
pip3 install -r requirements.txt
```

## 🏃‍♂️ Running the Application

### 1. Start Ollama (if not already running)

```bash
ollama serve
```

### 2. Start the Backend API

```bash
cd backend
python3 app.py
```

The API will be available at: `http://localhost:5000`

### 3. Start the Frontend (in a new terminal)

```bash
cd frontend
npm start
```

The React app will open at: `http://localhost:3000`

## 🎨 UI Features

### Input Form
- **Stock Ticker**: Enter any stock symbol (AAPL, MSFT, GOOGL, etc.)
- **Time Period**: Select analysis timeframe (1d, 1wk, 1mo, etc.)
- **Interval**: Choose data granularity (1m, 5m, 1h, 1d, etc.)
- **Risk Horizon**: Set risk analysis horizon in days

### Results Display
- **Summary Cards**: Key metrics at a glance
- **Detailed Report**: Full markdown-formatted analysis
- **Raw Data**: Collapsible JSON data for developers
- **Risk Flags**: Visual indicators for risk factors

### Visual Elements
- **Trend Icons**: Up/down arrows for price trends
- **Color Coding**: Green/red for positive/negative returns
- **Loading States**: Animated spinners during analysis
- **Responsive Design**: Works on desktop and mobile

## 🔧 API Endpoints

### POST `/api/analyze`
Analyze a stock using the multi-agent system.

**Request Body:**
```json
{
  "ticker": "AAPL",
  "period": "1wk",
  "interval": "1d",
  "horizon_days": 30
}
```

**Response:**
```json
{
  "ticker": "AAPL",
  "valuation": { ... },
  "sentiment": { ... },
  "metrics": { ... },
  "report": { ... }
}
```

### GET `/api/health`
Health check endpoint.

### GET `/api/models`
Get available models information.

## 🎯 Usage Examples

### Basic Stock Analysis
1. Enter stock ticker (e.g., "AAPL")
2. Select time period (e.g., "1wk")
3. Choose interval (e.g., "1d")
4. Set risk horizon (e.g., 30 days)
5. Click "Analyze Stock Risk"

### Advanced Analysis
- Try different time periods for various insights
- Use shorter intervals for intraday analysis
- Adjust risk horizon based on investment timeframe

## 🛠️ Customization

### Styling
- Modify `frontend/src/App.css` for custom styles
- Uses Tailwind CSS for rapid styling
- Responsive design with mobile-first approach

### API Integration
- Backend API in `backend/app.py`
- Easy to extend with additional endpoints
- CORS enabled for frontend communication

### Model Configuration
- Model settings in `src/config.py`
- Currently configured for Ollama Qwen:4b
- Easy to switch to other models

## 🔍 Troubleshooting

### Common Issues

1. **"Ollama call failed"**
   - Ensure Ollama is running: `ollama serve`
   - Check if qwen:4b model is installed: `ollama list`

2. **"Module not found" errors**
   - Run the setup script: `./setup.sh`
   - Or manually install dependencies

3. **CORS errors**
   - Backend should be running on port 5000
   - Frontend should be running on port 3000
   - Check browser console for detailed errors

4. **API connection failed**
   - Verify backend is running: `curl http://localhost:5000/api/health`
   - Check firewall settings

### Debug Mode

Enable debug mode in the backend:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 📊 Sample Output

The UI displays comprehensive analysis including:

- **Valuation Metrics**: Returns, volatility, trends
- **Risk Assessment**: VaR, drawdown, Sharpe ratio
- **Sentiment Analysis**: News sentiment and confidence
- **Investment Recommendations**: AI-generated insights
- **Risk Flags**: Automated risk warnings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Multi-Agent Finance Risk Analysis system.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the console logs
3. Ensure all dependencies are installed
4. Verify Ollama is running with qwen:4b model

---

**Happy Analyzing! 📈🤖**
