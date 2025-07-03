#!/bin/bash

# Script to set up and run the Difference Sort Streamlit app

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Difference Sort Setup Script ===${NC}"

# Navigate to the DifferenceSort directory
cd "$(dirname "$0")/DifferenceSort" || { echo -e "${RED}Error: DifferenceSort directory not found${NC}"; exit 1; }

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

# Ask user which version to run
echo -e "\n${GREEN}=== SELECT APPLICATION MODE ===${NC}"
echo -e "${YELLOW}1. Standard Demo Application${NC}"
echo -e "   - Interactive sorting demonstration"
echo -e "   - Performance comparisons"
echo -e "   - Step-by-step visualization\n"

echo -e "${YELLOW}2. Interactive Research Paper${NC}"
echo -e "   - Academic paper format"
echo -e "   - Mathematical proofs and theorems"
echo -e "   - Publication-ready presentation\n"

echo -e "${YELLOW}3. Launch Menu (Python-based selector)${NC}"
echo -e "   - Use the Python launcher for selection\n"

read -p "Enter your choice (1, 2, or 3): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting Standard Demo Application...${NC}"
        echo -e "${YELLOW}The app will open in your default browser at http://localhost:8501${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"
        streamlit run app.py
        ;;
    2)
        echo -e "\n${GREEN}Starting Interactive Research Paper...${NC}"
        echo -e "${YELLOW}The app will open in your default browser at http://localhost:8501${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}\n"
        streamlit run research_paper_app.py
        ;;
    3)
        echo -e "\n${GREEN}Launching Python-based menu...${NC}"
        $PYTHON_CMD run_app.py
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run the script again and select 1, 2, or 3.${NC}"
        exit 1
        ;;
esac