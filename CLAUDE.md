# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Running the Application
```bash
cd DifferenceSort
streamlit run app.py
```

### Installing Dependencies
```bash
cd DifferenceSort
pip install -r requirements.txt
# or using uv (if available):
uv pip install -r pyproject.toml
```

### Key Dependencies
- streamlit (UI framework)
- matplotlib, plotly (visualization)
- numpy, pandas (data processing)

## Architecture Overview

This is a Streamlit-based web application for demonstrating and analyzing custom sorting algorithms. The codebase follows a modular architecture with clear separation of concerns:

### Core Components

1. **app.py**: Main Streamlit application that orchestrates the UI and user interactions. Handles input methods, algorithm selection, step-by-step visualization, and performance comparisons.

2. **sorting_algorithms.py**: Contains three main classes:
   - `ReferenceBasedSorting`: Original algorithm that uses first element as reference, calculates differences, and positions elements
   - `OptimizedReferenceBasedSorting`: Hash-based optimization that eliminates null positions and handles sparse data efficiently
   - `StandardSorting`: Traditional sorting algorithms (bubble, quick, merge) for comparison

3. **visualization.py**: `SortingVisualizer` class creates interactive Plotly charts for each step of the sorting process, including positioning visualization and performance comparisons.

### Algorithm Flow

The reference-based sorting algorithm:
1. Selects first element as reference point
2. Calculates differences of all elements relative to reference
3. Positions elements based on their differences (original uses array with nulls, optimized uses hash map)
4. Assembles final sorted array

### Key Design Patterns

- **Step-by-step execution**: Both sorting implementations provide `sort_with_steps()` methods that return detailed information for visualization
- **Performance analysis**: Built-in timing and space complexity comparisons between algorithms
- **Interactive visualization**: Each algorithm step can be visualized using Plotly charts
- **Modular structure**: Clear separation between algorithms, visualization, and UI logic