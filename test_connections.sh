#!/bin/bash
set -e

echo "🔎 Checking services..."

check_service() {
  local name=$1
  local url=$2
  local match=$3
  echo -n "→ $name: "
  if curl -s "$url" | grep -q "$match"; then
    echo "✅ OK"
  else
    echo "❌ DOWN ($url)"
  fi
}

check_service "Backend"  "http://localhost:5001/api/health" "healthy"
check_service "Ollama"   "http://localhost:11434/api/tags"  "models"
check_service "Frontend" "http://localhost:3000"            "root"

echo "🚀 Running test analysis..."
curl -s -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL","period":"1d","interval":"1h","horizon_days":30}' \
  -w "\nHTTP Status: %{http_code}\n"