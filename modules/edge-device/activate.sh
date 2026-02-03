#!/bin/bash
#
# Activate the Python virtual environment for edge-device development.
#
# Usage:
#   source activate.sh
#
# Note: This script must be sourced, not executed directly.
#       Running ./activate.sh will NOT activate the venv in your shell.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/../../venv"

# Check if being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: This script must be sourced, not executed."
    echo ""
    echo "Usage:"
    echo "  source activate.sh"
    echo "  # or"
    echo "  . activate.sh"
    exit 1
fi

# Check if venv exists
if [ ! -f "$VENV_PATH/bin/activate" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo ""
    echo "Create it with:"
    echo "  python3 -m venv $VENV_PATH"
    echo "  source $VENV_PATH/bin/activate"
    echo "  pip install -r requirements.txt"
    return 1
fi

# Activate
source "$VENV_PATH/bin/activate"

echo "Virtual environment activated: $VENV_PATH"
echo "Python: $(which python)"
echo "Working directory: $SCRIPT_DIR"
