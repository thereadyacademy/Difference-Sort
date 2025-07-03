#!/bin/bash

# Script to set up and run the Integer Sorter Streamlit app

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Integer Sorter Setup Script ===${NC}"

# Navigate to the IntegerSorter directory
cd "$(dirname "$0")/IntegerSorter" || { echo -e "${RED}Error: IntegerSorter directory not found${NC}"; exit 1; }

# Check if Python is installed
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}Error: Python is not installed. Please install Python 3.7 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}Found Python: $($PYTHON_CMD --version)${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "\n${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate 2>/dev/null || venv\\Scripts\\activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    # Install from pyproject.toml dependencies
    pip install "matplotlib>=3.10.3" "numpy>=2.3.1" "pandas>=2.3.0" "plotly>=6.2.0" "streamlit>=1.46.1"
fi

echo -e "${GREEN}Dependencies installed successfully${NC}"

# Run the Streamlit app
echo -e "\n${GREEN}Starting Streamlit app...${NC}"
echo -e "${YELLOW}The app will open in your default browser at http://localhost:8501${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"

streamlit run app.py