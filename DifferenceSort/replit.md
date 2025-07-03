# Reference-Based Sorting Algorithm Tester

## Overview

This is a Streamlit web application that implements, tests, and visualizes a custom reference-based sorting algorithm. The application allows users to input data through multiple methods, compare the custom algorithm with standard sorting methods, and visualize the sorting process through interactive charts.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - Python-based web framework for rapid data app development
- **Visualization**: Plotly for interactive charts and graphs, Matplotlib for additional plotting capabilities
- **UI Components**: Streamlit widgets for input controls, configuration panels, and result display

### Backend Architecture
- **Language**: Python
- **Core Logic**: Custom sorting algorithm implementation with step-by-step execution tracking
- **Data Processing**: NumPy and Pandas for array manipulation and data handling
- **Modular Design**: Separated concerns with dedicated modules for algorithms and visualization

## Key Components

### 1. Main Application (`app.py`)
- **Purpose**: Entry point and UI orchestration
- **Key Features**:
  - Multiple input methods (manual, random generation, predefined test cases)
  - Interactive configuration panel
  - Real-time algorithm execution and comparison
- **Technologies**: Streamlit, NumPy, Pandas, Matplotlib, Plotly

### 2. Sorting Algorithms (`sorting_algorithms.py`)
- **Purpose**: Core algorithm implementations
- **Components**:
  - `ReferenceBasedSorting`: Custom algorithm using first element as reference
  - `StandardSorting`: Traditional sorting methods for comparison
- **Algorithm Steps**:
  1. Use first element as reference point
  2. Calculate differences relative to reference
  3. Position elements based on differences
  4. Remove null values for final sorted array

### 3. Visualization Engine (`visualization.py`)
- **Purpose**: Interactive visualization of sorting processes
- **Components**:
  - `SortingVisualizer`: Main visualization class
  - Positioning visualization for reference-based algorithm
  - Step-by-step algorithm execution display
- **Technologies**: Plotly for interactive charts

## Data Flow

1. **Input Stage**: User provides data through selected input method
2. **Processing Stage**: 
   - Data validation and parsing
   - Algorithm execution with step tracking
   - Performance metrics collection
3. **Visualization Stage**:
   - Generate interactive charts showing algorithm steps
   - Display comparison metrics
   - Render final results

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib**: Static plotting and visualization
- **Plotly**: Interactive charting and dashboards

### Rationale for Technology Choices
- **Streamlit**: Chosen for rapid prototyping and easy deployment of data applications
- **Plotly**: Selected for interactive visualizations that enhance user understanding of sorting steps
- **Modular Architecture**: Enables easy testing, maintenance, and extension of sorting algorithms

## Deployment Strategy

- **Platform**: Designed for Replit deployment with Streamlit hosting
- **Environment**: Python-based with pip package management
- **Scalability**: Single-user application optimized for educational and demonstration purposes
- **Configuration**: Minimal setup required, dependencies managed through requirements specification

## Changelog

- July 03, 2025. Initial setup with reference-based sorting algorithm and visualization
- July 03, 2025. Added optimization analysis and optimized hash-based sorting implementation

## User Preferences

Preferred communication style: Simple, everyday language.