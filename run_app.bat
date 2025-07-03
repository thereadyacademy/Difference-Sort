@echo off
REM Script to set up and run the Difference Sort Streamlit app on Windows

echo === Difference Sort Setup Script ===

REM Navigate to the DifferenceSort directory
cd /d "%~dp0DifferenceSort" || (
    echo Error: DifferenceSort directory not found
    pause
    exit /b 1
)

REM Check if Python is installed
echo.
echo Checking Python installation...
where python >nul 2>nul
if %errorlevel%==0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>nul
    if %errorlevel%==0 (
        set PYTHON_CMD=python3
    ) else (
        echo Error: Python is not installed. Please install Python 3.7 or higher.
        pause
        exit /b 1
    )
)

for /f "tokens=*" %%i in ('%PYTHON_CMD% --version') do echo Found: %%i

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    echo Virtual environment created
) else (
    echo.
    echo Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo.
echo Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    REM Install from pyproject.toml dependencies
    pip install "matplotlib>=3.10.3" "numpy>=2.3.1" "pandas>=2.3.0" "plotly>=6.2.0" "streamlit>=1.46.1"
)

echo Dependencies installed successfully

REM Ask user which version to run
echo.
echo === SELECT APPLICATION MODE ===
echo.
echo 1. Standard Demo Application
echo    - Interactive sorting demonstration
echo    - Performance comparisons
echo    - Step-by-step visualization
echo.
echo 2. Interactive Research Paper
echo    - Academic paper format
echo    - Mathematical proofs and theorems
echo    - Publication-ready presentation
echo.
echo 3. Launch Menu (Python-based selector)
echo    - Use the Python launcher for selection
echo.

set /p choice="Enter your choice (1, 2, or 3): "

if "%choice%"=="1" (
    echo.
    echo Starting Standard Demo Application...
    echo The app will open in your default browser at http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    streamlit run app.py
) else if "%choice%"=="2" (
    echo.
    echo Starting Interactive Research Paper...
    echo The app will open in your default browser at http://localhost:8501
    echo Press Ctrl+C to stop the server
    echo.
    streamlit run research_paper_app.py
) else if "%choice%"=="3" (
    echo.
    echo Launching Python-based menu...
    %PYTHON_CMD% run_app.py
) else (
    echo Invalid choice. Please run the script again and select 1, 2, or 3.
    pause
    exit /b 1
)