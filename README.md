# Difference Sort - Reference-Based Sorting Algorithm Visualizer

An interactive Streamlit application for demonstrating, analyzing, and comparing custom reference-based sorting algorithms with traditional sorting methods. This educational tool provides step-by-step visualization and performance analysis of innovative sorting approaches. Now includes an interactive research paper format for academic presentation.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Explanation](#algorithm-explanation)
- [Architecture](#architecture)
- [Performance Analysis](#performance-analysis)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Overview

Difference Sort is a web-based application that implements and visualizes a novel reference-based sorting algorithm. The algorithm uses the concept of differences from a reference element to determine the sorted positions of array elements. The project includes comprehensive visualization and performance comparison tools, now available in both standard demo and interactive research paper formats.

### What makes this project unique?
- **Novel Algorithm**: Implements a unique sorting approach based on element differences
- **Interactive Visualization**: Step-by-step visual representation of the sorting process
- **Performance Analysis**: Real-time comparison with standard sorting algorithms
- **Educational Value**: Perfect for understanding algorithm complexity and optimization techniques
- **Research Paper Format**: Interactive academic paper presentation with proofs and theorems

## Key Features

### 🔧 Core Functionality
- **Multiple Input Methods**
  - Manual input (comma or space-separated)
  - Random array generation with customizable size and range
  - Predefined test cases for edge cases and common scenarios

- **Algorithm Implementations**
  - Reference-Based Sorting (with positioned array and null padding)
  - Standard algorithms for comparison (Bubble, Quick, Merge, Built-in)

- **Visualization Features**
  - Step-by-step algorithm execution
  - Interactive Plotly charts
  - Positioning visualization with null value representation
  - Before/after array transformation views

- **Performance Analysis**
  - Time complexity comparison across different array sizes (up to 10M elements)
  - Space efficiency analysis for sparse data
  - Real-time execution benchmarks
  - Logarithmic scale performance charts

- **Edge Case Testing**
  - Empty arrays
  - Single element arrays
  - Arrays with duplicates
  - Negative numbers
  - Mixed positive/negative values

### 📚 Research Paper Mode
- **Academic Format**: Full research paper structure with abstract, introduction, and conclusion
- **Mathematical Proofs**: Formal complexity proofs with LaTeX-style equations
- **Interactive Theorems**: Embedded demonstrations within the paper
- **Publication Quality**: Professional formatting suitable for academic submission
- **Bibliography**: Complete references and citations

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/difference-sort.git
   cd difference-sort
   ```

2. **Run the setup script**
   
   The project includes convenient setup scripts for all platforms:
   
   **Linux/Mac:**
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```
   
   **Windows:**
   ```cmd
   run_app.bat
   ```
   
   **Cross-platform Python:**
   ```bash
   python run_app.py
   ```

   These scripts will automatically:
   - Create a virtual environment
   - Install all dependencies
   - Launch the Streamlit application

### Manual Installation

If you prefer manual installation:

```bash
# Navigate to the application directory
cd DifferenceSort

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Usage

### Starting the Application
Once installed, you have two options:

#### Option 1: Standard Demo Application
```bash
streamlit run app.py
```
The standard demo will open in your browser at `http://localhost:8501`

#### Option 2: Interactive Research Paper
```bash
streamlit run research_paper_app.py
```
The research paper version will open in your browser with academic formatting

#### Option 3: Use the Launcher (Recommended)
```bash
python run_app.py
```
This will present you with a menu to choose between the standard demo and research paper formats

### Input Methods

1. **Manual Input**
   - Enter integers separated by commas or spaces
   - Example: `5, 2, 8, 1, 9` or `5 2 8 1 9`

2. **Random Generation**
   - Use sliders to set array size (5-50 elements)
   - Set maximum value range (10-1000)
   - Click "Generate Random Array"

3. **Predefined Test Cases**
   - Small Array: `[5, 2, 8, 1, 9]`
   - Already Sorted: `[1, 2, 3, 4, 5]`
   - Reverse Sorted: `[5, 4, 3, 2, 1]`
   - With Duplicates: `[3, 1, 4, 1, 5, 9, 2, 6]`
   - Large Range: `[100, 5, 200, 15, 50]`
   - Single Element: `[42]`
   - Two Elements: `[3, 1]`


### Performance Testing
- Click "Run Performance Tests" to benchmark algorithms with arrays up to 10M elements
- Click "Test Space Efficiency" to analyze memory usage with sparse data

## Algorithm Explanation

### Original Reference-Based Sorting

The algorithm works in four main steps:

1. **Reference Selection**: The first element becomes the reference point
2. **Difference Calculation**: Calculate the difference of each element relative to the reference
3. **Positioning**: Place elements in positions based on their differences
4. **Null Removal**: Remove null values to get the final sorted array

#### Example Walkthrough
For array `[5, 2, 8, 1, 9]`:
- **Reference**: 5
- **Differences**: [0, -3, 3, -4, 4]
- **Min difference**: -4, Max difference: 4
- **Position mapping**: 
  - 1 (diff=-4) → position 0
  - 2 (diff=-3) → position 1
  - 5 (diff=0) → position 4
  - 8 (diff=3) → position 7
  - 9 (diff=4) → position 8
- **Positioned array**: [1, 2, null, null, 5, null, null, 8, 9]
- **Final sorted**: [1, 2, 5, 8, 9]


## Architecture

The application follows a modular architecture with clear separation of concerns:

```
DifferenceSort/
├── app.py                 # Main Streamlit application
├── research_paper_app.py  # Interactive research paper version
├── run_app.py            # Launcher script for both versions
├── sorting_algorithms.py  # Algorithm implementations
├── visualization.py       # Visualization utilities
├── requirements.txt       # Python dependencies
└── pyproject.toml        # Project configuration
```

### Core Components

#### `app.py`
- Main application orchestrator
- Handles user interface and interactions
- Manages input validation and processing
- Coordinates algorithm execution and visualization

#### `research_paper_app.py`
- Interactive research paper format
- Academic paper structure with sections
- Mathematical proofs and theorems
- LaTeX-style equations and formatting
- Embedded interactive demonstrations
- Bibliography and citations

#### `sorting_algorithms.py`
Contains two main classes:
- **`ReferenceBasedSorting`**: Original algorithm implementation with positioned array
- **`StandardSorting`**: Traditional algorithms for comparison (bubble, quick, merge)

#### `visualization.py`
- **`SortingVisualizer`**: Creates interactive Plotly charts
- Provides step-by-step visualization methods
- Handles positioning visualization with null representation
- Creates performance comparison charts

## Performance Analysis

### Time Complexity
| Algorithm | Best Case | Average Case | Worst Case | Space Complexity |
|-----------|-----------|--------------|------------|------------------|
| Reference-Based | O(n) | O(n + k) | O(n + k) | O(k) |
| Built-in Sort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Bubble Sort | O(n) | O(n²) | O(n²) | O(1) |
| Quick Sort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Merge Sort | O(n log n) | O(n log n) | O(n log n) | O(n) |

*Note: k = range of input values (max - min + 1), n = number of elements*

### Space Efficiency

The reference-based algorithm's space complexity depends on the range of values:
- Dense data (k ≈ n): Efficient space usage
- Sparse data (k >> n): High memory usage due to null positions
- Example: Array [1, 1000] requires 1000 positions despite having only 2 elements

## Examples

### Basic Usage Example
```python
# Input array
arr = [5, 2, 8, 1, 9]

# Reference-based algorithm result
# Step 1: Reference = 5
# Step 2: Differences = [0, -3, 3, -4, 4]
# Step 3: Positioned = [1, 2, null, null, 5, null, null, 8, 9]
# Step 4: Sorted = [1, 2, 5, 8, 9]
```

### Edge Cases Handled
- Empty arrays: Returns empty array
- Single element: Returns unchanged
- Duplicates: Maintains relative order
- Negative numbers: Handles correctly
- Mixed positive/negative: Sorts properly

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add type hints for function parameters
- Include docstrings for all classes and methods
- Write unit tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Visualizations powered by [Plotly](https://plotly.com/)
- Developed as part of The READY Academy curriculum

## Future Enhancements

- [ ] Add more sorting algorithm variants
- [ ] Implement parallel processing for large arrays
- [ ] Add export functionality for visualizations
- [ ] Create API endpoints for algorithm access
- [ ] Add more comprehensive unit tests
- [ ] Support for custom comparison functions
- [ ] Mobile-responsive design improvements

---

For questions or support, please open an issue on the GitHub repository.