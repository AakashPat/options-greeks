#!/bin/bash

# Setup script for Options Greeks Calculator & Gamma Scalping Backtester
# This script automates the setup process on Unix-like systems

set -e  # Exit on any error

echo "🚀 Setting up Options Greeks Calculator & Gamma Scalping Backtester"
echo "=================================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "   Please install Python 3.8 or higher"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment (recommended)
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Setup environment file
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✅ Created .env from env.example"
    else
        # Create basic .env file
        cat > .env << 'EOF'
# FutPrint API Configuration
# Replace these with your actual API credentials

API_OHLCV_URL=https://api.futprint.in/api/historical-data-ohlcv-oi
API_TOKEN=<YOUR_TOKEN>
API_KEY=FutPrintIN
RISK_FREE_RATE=0.06
DEFAULT_LOT_SIZE=50
DEFAULT_FEE_PER_CONTRACT=10.0
DEFAULT_SLIPPAGE_TICKS=1
DEFAULT_TICK_SIZE=1.0
EOF
        echo "✅ Created basic .env file"
    fi
    echo "   Please edit .env with your actual API credentials"
else
    echo "⚠️  .env file already exists, skipping..."
fi

# Test imports
echo "🔍 Verifying installation..."
python3 -c "
import pandas as pd
import numpy as np
import requests
import scipy
import matplotlib
from dotenv import load_dotenv
print('✅ All packages imported successfully')
"

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your FutPrint API credentials:"
echo "   nano .env"
echo ""
echo "2. To activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the application:"
echo "   python main.py"
echo ""
echo "For more information, see README.md"
