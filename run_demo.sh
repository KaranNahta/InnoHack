#!/usr/bin/env bash
# ==============================================================================
# CASPER-Gov: 1-Click Unified Demo Launcher
# ==============================================================================
# Verifies environment, builds model artifacts if missing, and launches both:
#   1. FastAPI ML & Enforcement Engine (http://localhost:8000/docs)
#   2. Streamlit Interactive Command Center (http://localhost:8501)
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================================"
echo "   🌾 CASPER-Gov: Essential Commodity Surveillance & Price Enforcement"
echo "========================================================================"

# Determine Python environment
if [ -d ".venv" ]; then
    PYTHON_EXEC="./.venv/bin/python"
    UVICORN_EXEC="./.venv/bin/uvicorn"
    STREAMLIT_EXEC="./.venv/bin/streamlit"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="python3"
    UVICORN_EXEC="uvicorn"
    STREAMLIT_EXEC="streamlit"
else
    PYTHON_EXEC="python"
    UVICORN_EXEC="uvicorn"
    STREAMLIT_EXEC="streamlit"
fi

echo "[1/4] Checking Python environment: $PYTHON_EXEC"

# Check if model artifacts exist; if not, train them
if [ ! -f "models/lgb_p50.joblib" ] || [ ! -f "models/mapie_conformal.joblib" ] || [ ! -f "models/pca_macro.joblib" ]; then
    echo "[2/4] Model artifacts missing. Running 1-command training pipeline..."
    PYTHONPATH=. $PYTHON_EXEC scripts/train_all_models.py
else
    echo "[2/4] Verified trained model artifacts in models/"
fi

# Cleanup existing background processes on exit
cleanup() {
    echo ""
    echo "Shutting down CASPER-Gov services..."
    kill $(jobs -p) 2>/dev/null || true
    echo "Services stopped cleanly."
}
trap cleanup EXIT INT TERM

echo "[3/4] Launching FastAPI Backend Engine on http://0.0.0.0:8000..."
PYTHONPATH=. $UVICORN_EXEC src.api.main:app --host 0.0.0.0 --port 8000 --log-level warning &
BACKEND_PID=$!

# Wait briefly for backend to warm up
sleep 2

echo "[4/4] Launching Streamlit Interactive Command Center on http://0.0.0.0:8501..."
echo "========================================================================"
echo "   🚀 CASPER-Gov Live Demo Active:"
echo "      • Streamlit Dashboard:  http://localhost:8501"
echo "      • FastAPI Swagger Docs: http://localhost:8000/docs"
echo "      • Press Ctrl+C to terminate all services."
echo "========================================================================"

PYTHONPATH=. $STREAMLIT_EXEC run src/dashboard/app.py --server.port 8501 --server.headless false

wait
