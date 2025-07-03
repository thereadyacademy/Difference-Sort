#!/usr/bin/env python3
"""
Launcher script for the Difference Sort application.
Allows users to choose between the standard demo and the interactive research paper.
"""

import sys
import subprocess

def main():
    print("\n" + "="*60)
    print("REFERENCE-BASED SORTING ALGORITHM")
    print("="*60)
    print("\nPlease select which version to run:\n")
    print("1. Standard Demo Application")
    print("   - Interactive sorting demonstration")
    print("   - Performance comparisons")
    print("   - Step-by-step visualization\n")
    
    print("2. Interactive Research Paper")
    print("   - Academic paper format")
    print("   - Mathematical proofs and theorems")
    print("   - Interactive demonstrations")
    print("   - Publication-ready presentation\n")
    
    print("3. Exit\n")
    
    while True:
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            print("\nLaunching Standard Demo Application...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
            break
        elif choice == '2':
            print("\nLaunching Interactive Research Paper...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "research_paper_app.py"])
            break
        elif choice == '3':
            print("\nExiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()