#!/bin/bash
# Pipeline Data Portal - Start Script
# Starts both the backend API and frontend dev server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
CERULEAN='\033[38;5;39m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${CERULEAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           Pipeline Data Portal v1.0                      ║"
echo "║           Dark Grey + Cerulean Blue                      ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if backend dependencies are installed
echo -e "${YELLOW}[1/4] Checking backend dependencies...${NC}"
cd "$SCRIPT_DIR/backend"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing backend dependencies..."
    pip install -r requirements.txt
fi
echo -e "${GREEN}✓ Backend dependencies ready${NC}"

# Check if frontend dependencies are installed
echo -e "${YELLOW}[2/4] Checking frontend dependencies...${NC}"
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi
echo -e "${GREEN}✓ Frontend dependencies ready${NC}"

# Start backend
echo -e "${YELLOW}[3/4] Starting backend server on port 8088...${NC}"
cd "$SCRIPT_DIR/backend"
python3 main.py &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Give backend time to start
sleep 2

# Start frontend
echo -e "${YELLOW}[4/4] Starting frontend dev server on port 3000...${NC}"
cd "$SCRIPT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

echo ""
echo -e "${CERULEAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Portal is running!${NC}"
echo ""
echo -e "  Frontend: ${CERULEAN}http://localhost:3000${NC}"
echo -e "  Backend:  ${CERULEAN}http://localhost:8088${NC}"
echo -e "  API Docs: ${CERULEAN}http://localhost:8088/docs${NC}"
echo ""
echo -e "Press Ctrl+C to stop both servers"
echo -e "${CERULEAN}══════════════════════════════════════════════════════════${NC}"

# Trap cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Portal stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for both processes
wait
