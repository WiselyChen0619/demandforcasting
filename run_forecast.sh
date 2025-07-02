#!/bin/bash
# Demand Forecasting System Execution Script

echo "=== Demand Forecasting System ==="
echo "Starting at: $(date)"
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
echo "Setting up directories..."
mkdir -p data models output docs

# Run the forecasting system
echo
echo "Running demand forecasting..."
cd src
python main.py "$@"

# Return to root directory
cd ..

echo
echo "Forecasting complete at: $(date)"
echo "Results available in the output/ directory"

# Deactivate virtual environment
deactivate