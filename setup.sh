#!/bin/bash

# Finance Risk Analysis UI Setup Script

echo "🚀 Setting up Finance Risk Analysis UI..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd frontend
npm install

if [ $? -eq 0 ]; then
    echo "✅ Frontend dependencies installed successfully"
else
    echo "❌ Failed to install frontend dependencies"
    exit 1
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd ../backend
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Backend dependencies installed successfully"
else
    echo "❌ Failed to install backend dependencies"
    exit 1
fi

# Install main project dependencies
echo "📦 Installing main project dependencies..."
cd ..
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Main project dependencies installed successfully"
else
    echo "❌ Failed to install main project dependencies"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 To run the application:"
echo "   1. Start the backend API:"
echo "      cd backend && python3 app.py"
echo ""
echo "   2. In a new terminal, start the frontend:"
echo "      cd frontend && npm start"
echo ""
echo "   3. Open your browser to: http://localhost:3000"
echo ""
echo "🤖 Make sure Ollama with qwen:4b is running locally!"
echo "   Run: ollama serve (in another terminal if not already running)"
