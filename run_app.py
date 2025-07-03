#!/usr/bin/env python3
"""
Cross-platform script to set up and run the Difference Sort Streamlit app
"""

import os
import sys
import subprocess
import platform

# Colors for terminal output (works on most terminals)
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    @staticmethod
    def disable():
        Colors.GREEN = ''
        Colors.YELLOW = ''
        Colors.RED = ''
        Colors.RESET = ''

# Disable colors on Windows if not supported
if platform.system() == 'Windows' and not os.environ.get('ANSICON'):
    Colors.disable()

def print_colored(message, color=Colors.RESET):
    print(f"{color}{message}{Colors.RESET}")

def run_command(command, shell=True):
    """Run a command and return success status"""
    try:
        subprocess.run(command, shell=shell, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print_colored("=== Difference Sort Setup Script ===", Colors.GREEN)
    
    # Navigate to the DifferenceSort directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(script_dir, "DifferenceSort")
    
    if not os.path.exists(app_dir):
        print_colored("Error: DifferenceSort directory not found", Colors.RED)
        sys.exit(1)
    
    os.chdir(app_dir)
    
    # Check Python version
    print_colored("\nChecking Python version...", Colors.YELLOW)
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print_colored("Error: Python 3.7 or higher is required", Colors.RED)
        sys.exit(1)
    print_colored(f"Found Python {python_version.major}.{python_version.minor}.{python_version.micro}", Colors.GREEN)
    
    # Create virtual environment if it doesn't exist
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print_colored("\nCreating virtual environment...", Colors.YELLOW)
        subprocess.run([sys.executable, "-m", "venv", venv_path])
        print_colored("Virtual environment created", Colors.GREEN)
    else:
        print_colored("\nVirtual environment already exists", Colors.GREEN)
    
    # Determine the correct pip executable
    if platform.system() == "Windows":
        pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
        python_exe = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        pip_exe = os.path.join(venv_path, "bin", "pip")
        python_exe = os.path.join(venv_path, "bin", "python")
    
    # Upgrade pip
    print_colored("\nUpgrading pip...", Colors.YELLOW)
    subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
    
    # Install dependencies
    print_colored("\nInstalling dependencies...", Colors.YELLOW)
    requirements_file = "requirements.txt"
    
    if os.path.exists(requirements_file):
        subprocess.run([pip_exe, "install", "-r", requirements_file])
    else:
        # Install from pyproject.toml dependencies
        packages = [
            "matplotlib>=3.10.3",
            "numpy>=2.3.1",
            "pandas>=2.3.0",
            "plotly>=6.2.0",
            "streamlit>=1.46.1"
        ]
        subprocess.run([pip_exe, "install"] + packages)
    
    print_colored("Dependencies installed successfully", Colors.GREEN)
    
    # Ask user which version to run
    print_colored("\n=== SELECT APPLICATION MODE ===", Colors.GREEN)
    print("\n1. Standard Demo Application")
    print("   - Interactive sorting demonstration")
    print("   - Performance comparisons")
    print("   - Step-by-step visualization\n")
    
    print("2. Interactive Research Paper")
    print("   - Academic paper format")
    print("   - Mathematical proofs and theorems")
    print("   - Publication-ready presentation\n")
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            app_file = "app.py"
            print_colored("\nStarting Standard Demo Application...", Colors.GREEN)
            break
        elif choice == '2':
            app_file = "research_paper_app.py"
            print_colored("\nStarting Interactive Research Paper...", Colors.GREEN)
            break
        else:
            print_colored("Invalid choice. Please enter 1 or 2.", Colors.RED)
    
    print_colored("The app will open in your default browser at http://localhost:8501", Colors.YELLOW)
    print_colored("Press Ctrl+C to stop the server\n", Colors.YELLOW)
    
    streamlit_exe = os.path.join(venv_path, "Scripts" if platform.system() == "Windows" else "bin", "streamlit")
    
    try:
        subprocess.run([streamlit_exe, "run", app_file])
    except KeyboardInterrupt:
        print_colored("\n\nStreamlit server stopped", Colors.YELLOW)

if __name__ == "__main__":
    main()