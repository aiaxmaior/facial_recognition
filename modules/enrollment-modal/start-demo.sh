#!/bin/bash
#
# Start Enrollment Modal Demo
# Launches both Python API (DeepFace) and Node.js frontend
#
# Usage: ./start-demo.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Conda environment name
CONDA_ENV="facial_recog"

# Ports
PYTHON_PORT=5000
NODE_PORT=3001
VITE_PORT=5173

# PIDs for cleanup
PYTHON_PID=""
NODE_PID=""

cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    
    # Kill by port to ensure cleanup (more reliable than PID tracking with conda)
    for port in $PYTHON_PORT $NODE_PORT $VITE_PORT; do
        pid=$(lsof -ti:$port 2>/dev/null)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null && echo "Stopped process on port $port (PID $pid)"
        fi
    done
    
    if [ -n "$NODE_PID" ]; then
        kill $NODE_PID 2>/dev/null
        pkill -P $NODE_PID 2>/dev/null
    fi
    
    echo -e "${GREEN}Cleanup complete${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    Enrollment Modal Demo Launcher      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Kill any existing processes on our ports
echo -e "${YELLOW}Cleaning up any existing processes...${NC}"
for port in $PYTHON_PORT $NODE_PORT $VITE_PORT; do
    pid=$(lsof -ti:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo "  Killing process on port $port (PID $pid)"
        kill -9 $pid 2>/dev/null
    fi
done
sleep 1

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda/Miniconda."
    exit 1
fi

# Check if conda env exists
if ! conda env list 2>/dev/null | grep -q "${CONDA_ENV}"; then
    echo "Error: conda environment '${CONDA_ENV}' not found."
    echo "Create it with: conda create -n ${CONDA_ENV} python=3.10"
    exit 1
fi

# Start Python API
echo -e "${GREEN}Starting Python API (DeepFace)...${NC}"
conda run -n "$CONDA_ENV" --no-capture-output \
    python python-api/enrollment_api.py &
PYTHON_PID=$!

# Wait for Python API to be ready
echo -n "Waiting for Python API..."
for i in {1..30}; do
    if curl -s http://localhost:$PYTHON_PORT/health > /dev/null 2>&1; then
        echo -e " ${GREEN}ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done

if ! curl -s http://localhost:$PYTHON_PORT/health > /dev/null 2>&1; then
    echo -e " ${YELLOW}timeout (continuing anyway)${NC}"
fi

# Start Node.js demo
echo -e "${GREEN}Starting Node.js demo...${NC}"
PORT=$NODE_PORT AUDIO_PATH="$SCRIPT_DIR/audio" npm run dev &
NODE_PID=$!

# Wait for services to start
sleep 3

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}Services running:${NC}"
echo -e "  Python API:  http://localhost:${PYTHON_PORT}"
echo -e "  Node.js API: http://localhost:${NODE_PORT}"
echo -e "  Demo UI:     ${GREEN}http://localhost:${VITE_PORT}${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait
