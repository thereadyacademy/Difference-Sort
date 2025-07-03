@echo off
REM Script to set up and run the Integer Sorter Streamlit app on Windows

echo === Integer Sorter Setup Script ===

REM Navigate to the IntegerSorter directory
cd /d "%~dp0IntegerSorter" || (
    echo Error: IntegerSorter directory not found
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

REM Run the Streamlit app
echo.
echo Starting Streamlit app...
echo The app will open in your default browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

streamlit run app.py