import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import multiprocessing
from sorting_algorithms import ReferenceBasedSorting, StandardSorting, OptimizedReferenceBasedSorting
from parallel_sorting_algorithms import (
    ParallelReferenceBasedSorting, 
    ParallelStandardSorting,
    OptimizedParallelReferenceBasedSorting
)
from multiprocess_sorting_algorithms import (
    MultiprocessReferenceBasedSorting,
    MultiprocessStandardSorting
)
from visualization import SortingVisualizer

def main():
    st.title("Reference-Based Sorting Algorithm Tester")
    st.markdown("### Test and visualize a custom reference-based sorting algorithm")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Thread control settings
    st.sidebar.subheader("Thread Control Settings")
    use_parallel = st.sidebar.checkbox("Enable Parallel Processing", value=False)
    
    max_threads = multiprocessing.cpu_count()
    if use_parallel:
        parallel_method = st.sidebar.radio(
            "Parallelization Method",
            ["Threading", "Multiprocessing"],
            help="Multiprocessing provides true parallelism but has more overhead"
        )
        
        num_threads = st.sidebar.slider(
            f"Number of {'Threads' if parallel_method == 'Threading' else 'Processes'}",
            min_value=1,
            max_value=max_threads * 2,
            value=max_threads,
            help=f"Your system has {max_threads} CPU cores"
        )
    else:
        num_threads = 1
        parallel_method = "Threading"
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm Selection")
    algorithm_type = st.sidebar.selectbox(
        "Choose Algorithm Type",
        ["Original Reference-Based", "Optimized Hash-Based", "Standard Algorithms"]
    )
    
    # Input section
    st.header("Input Data")
    
    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Manual Input", "Generate Random Array", "Predefined Test Cases"]
    )
    
    input_array = []
    
    if input_method == "Manual Input":
        user_input = st.text_input(
            "Enter integers (comma or space separated):",
            placeholder="e.g., 5,2,8,1,9 or 5 2 8 1 9"
        )
        
        if user_input:
            try:
                # Handle both comma and space separated values
                if ',' in user_input:
                    input_array = [int(x.strip()) for x in user_input.split(',')]
                else:
                    input_array = [int(x.strip()) for x in user_input.split()]
            except ValueError:
                st.error("Please enter valid integers separated by commas or spaces.")
                return
    
    elif input_method == "Generate Random Array":
        col1, col2 = st.columns(2)
        with col1:
            array_size = st.slider("Array Size:", 5, 50, 10)
        with col2:
            max_value = st.slider("Max Value:", 10, 1000, 100)
        
        if st.button("Generate Random Array"):
            input_array = np.random.randint(1, max_value + 1, array_size).tolist()
            st.success(f"Generated array: {input_array}")
    
    else:  # Predefined test cases
        test_cases = {
            "Small Array": [5, 2, 8, 1, 9],
            "Already Sorted": [1, 2, 3, 4, 5],
            "Reverse Sorted": [5, 4, 3, 2, 1],
            "Duplicates": [3, 1, 4, 1, 5, 9, 2, 6],
            "Large Range": [100, 5, 200, 15, 50],
            "Single Element": [42],
            "Two Elements": [3, 1]
        }
        
        selected_case = st.selectbox("Select test case:", list(test_cases.keys()))
        input_array = test_cases[selected_case]
        st.info(f"Selected array: {input_array}")
    
    # Algorithm explanation (available before input validation)
    with st.expander("Algorithm Explanation"):
        st.markdown("""
        **Reference-Based Sorting Algorithm:**
        1. **Reference Selection**: The first element becomes the reference point
        2. **Difference Calculation**: Calculate the difference of each element relative to the reference
        3. **Positioning**: Place elements in positions equal to their differences (with null padding)
        4. **Null Removal**: Remove null values to get the final sorted array
        
        **Example**: For array [5, 2, 8, 1, 9]
        - Reference: 5
        - Differences: [0, -3, 3, -4, 4]
        - Positioned array: [null, null, null, 1, null, 2, null, null, 5, 8, 9]
        - Final sorted: [1, 2, 5, 8, 9]
        """)
        
    
    # Validation and processing
    if not input_array:
        st.warning("Please provide an input array to proceed.")
        return
    
    if len(input_array) == 0:
        st.error("Array cannot be empty.")
        return
    
    # Display input array
    st.subheader("Input Array")
    st.write(f"Array: {input_array}")
    st.write(f"Size: {len(input_array)} elements")
    
    # Initialize sorting classes based on selection
    visualizer = SortingVisualizer()
    
    # Show thread info
    if use_parallel:
        st.info(f"🔧 Parallel processing enabled with {num_threads} {'threads' if parallel_method == 'Threading' else 'processes'} ({parallel_method})")
    
    # Select appropriate sorting class
    if algorithm_type == "Original Reference-Based":
        if use_parallel:
            if parallel_method == "Multiprocessing":
                ref_sorter = MultiprocessReferenceBasedSorting(num_threads)
            else:
                ref_sorter = ParallelReferenceBasedSorting(num_threads)
        else:
            ref_sorter = ReferenceBasedSorting()
        algorithm_name = "Reference-Based Sorting"
    elif algorithm_type == "Optimized Hash-Based":
        if use_parallel:
            if parallel_method == "Multiprocessing":
                # Use multiprocess version for optimized too
                ref_sorter = MultiprocessReferenceBasedSorting(num_threads)
            else:
                ref_sorter = OptimizedParallelReferenceBasedSorting(num_threads)
        else:
            ref_sorter = OptimizedReferenceBasedSorting(num_threads)
        algorithm_name = "Optimized Hash-Based Sorting"
    else:
        # Standard algorithms
        if use_parallel:
            if parallel_method == "Multiprocessing":
                ref_sorter = MultiprocessStandardSorting(num_threads)
            else:
                ref_sorter = ParallelStandardSorting(num_threads)
        else:
            ref_sorter = StandardSorting()
        algorithm_name = "Standard Sorting Algorithms"
    
    if use_parallel and parallel_method == "Multiprocessing":
        std_sorter = MultiprocessStandardSorting(num_threads)
    elif use_parallel:
        std_sorter = ParallelStandardSorting(num_threads)
    else:
        std_sorter = StandardSorting()
    
    # Execute sorting algorithm with steps
    st.header(f"{algorithm_name} Steps")
    try:
        if algorithm_type == "Standard Algorithms":
            # Standard algorithms don't have step-by-step visualization
            st.info("Standard algorithms don't support step-by-step visualization")
            sorted_result = sorted(input_array.copy())
            st.write(f"**Sorted Array**: {sorted_result}")
        else:
            steps = ref_sorter.sort_with_steps(input_array.copy())
            
            # Display each step
            for i, step in enumerate(steps):
                st.subheader(f"Step {i + 1}: {step['description']}")
                
                if step['type'] == 'reference':
                    st.write(f"**Reference Element**: {step['reference']}")
                    st.write(f"**Array**: {step['array']}")
                
                elif step['type'] == 'differences':
                    st.write(f"**Differences**: {step['differences']}")
                    # Display differences calculation
                    diff_df = pd.DataFrame({
                        'Element': step['array'],
                        'Reference': [step['reference']] * len(step['array']),
                        'Difference': step['differences']
                    })
                    st.table(diff_df)
                
                elif step['type'] == 'positioning':
                    st.write(f"**Positioned Array**: {step['positioned_array']}")
                    # Visualize positioning
                    fig = visualizer.visualize_positioning(
                        step['positioned_array'], 
                        step['reference'], 
                        step['differences']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                elif step['type'] == 'final':
                    st.write(f"**Final Sorted Array**: {step['sorted_array']}")
                    # Show before and after
                    comparison_df = pd.DataFrame({
                        'Original': input_array + [None] * (len(step['sorted_array']) - len(input_array)),
                        'Sorted': step['sorted_array'] + [None] * (len(input_array) - len(step['sorted_array']))
                    })
                    st.table(comparison_df)
        
    except Exception as e:
        st.error(f"Error in reference-based sorting: {str(e)}")
        return
    
    # Performance comparison
    st.header("Performance Comparison")
    
    # Time complexity analysis
    st.subheader("Time Complexity Analysis")
    complexity_data = {
        'Algorithm': ['Reference-Based', 'Built-in Sort', 'Bubble Sort', 'Quick Sort', 'Merge Sort'],
        'Best Case': ['O(n)', 'O(n log n)', 'O(n)', 'O(n log n)', 'O(n log n)'],
        'Average Case': ['O(n + k)', 'O(n log n)', 'O(n²)', 'O(n log n)', 'O(n log n)'],
        'Worst Case': ['O(n + k)', 'O(n log n)', 'O(n²)', 'O(n²)', 'O(n log n)'],
        'Space Complexity': ['O(k)', 'O(1)', 'O(1)', 'O(log n)', 'O(n)']
    }
    complexity_df = pd.DataFrame(complexity_data)
    st.table(complexity_df)
    st.caption("k = range of input values")
    
    # Execution time comparison
    st.subheader("Execution Time Comparison")
    
    # Test configuration
    col1, col2 = st.columns(2)
    with col1:
        test_parallel = st.checkbox("Include Parallel Tests", value=use_parallel)
    with col2:
        if test_parallel:
            thread_counts = st.multiselect(
                "Thread Counts to Test",
                options=[1, 2, 4, 8, 16, 32],
                default=[1, max_threads, max_threads * 2] if max_threads <= 8 else [1, 4, 8, max_threads]
            )
        else:
            thread_counts = [1]
    
    # Test with different array sizes
    sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    if st.button("Run Performance Tests"):
        st.warning("⚠️ Testing large arrays (up to 10M elements) may take several minutes to complete.")
        
        performance_data = {
            'Array Size': []
        }
        
        # Add columns for each algorithm and thread count
        if algorithm_type == "Original Reference-Based":
            performance_data['Reference-Based (ms)'] = []
            if test_parallel:
                for threads in thread_counts:
                    performance_data[f'Parallel Ref-Based ({threads}T)'] = []
        elif algorithm_type == "Optimized Hash-Based":
            performance_data['Optimized Hash-Based (ms)'] = []
            if test_parallel:
                for threads in thread_counts:
                    performance_data[f'Parallel Optimized ({threads}T)'] = []
        
        performance_data['Built-in Sort (ms)'] = []
        performance_data['Bubble Sort (ms)'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, size in enumerate(sizes):
            status_text.text(f'Testing array size: {size:,} ({i+1}/{len(sizes)} - {((i+1)/len(sizes)*100):.0f}%)')
            
            # Generate test array
            test_array = np.random.randint(1, 1000, size).tolist()
            
            # Test selected algorithm
            if algorithm_type == "Original Reference-Based":
                # Test sequential
                start_time = time.time()
                ReferenceBasedSorting().sort(test_array.copy())
                ref_time = (time.time() - start_time) * 1000
                performance_data['Reference-Based (ms)'].append(ref_time)
                
                # Test parallel versions
                if test_parallel:
                    for threads in thread_counts:
                        start_time = time.time()
                        ParallelReferenceBasedSorting(threads).sort(test_array.copy())
                        par_time = (time.time() - start_time) * 1000
                        performance_data[f'Parallel Ref-Based ({threads}T)'].append(par_time)
            
            elif algorithm_type == "Optimized Hash-Based":
                # Test sequential
                start_time = time.time()
                OptimizedReferenceBasedSorting().sort(test_array.copy())
                opt_time = (time.time() - start_time) * 1000
                performance_data['Optimized Hash-Based (ms)'].append(opt_time)
                
                # Test parallel versions
                if test_parallel:
                    for threads in thread_counts:
                        start_time = time.time()
                        OptimizedParallelReferenceBasedSorting(threads).sort(test_array.copy())
                        par_time = (time.time() - start_time) * 1000
                        performance_data[f'Parallel Optimized ({threads}T)'].append(par_time)
            
            
            # Test built-in sort
            start_time = time.time()
            std_sorter.builtin_sort(test_array.copy())
            builtin_time = (time.time() - start_time) * 1000
            
            # Test bubble sort (only for smaller arrays due to O(n²) complexity)
            if size <= 1000:
                start_time = time.time()
                std_sorter.bubble_sort(test_array.copy())
                bubble_time = (time.time() - start_time) * 1000
            else:
                bubble_time = None
            
            performance_data['Array Size'].append(size)
            performance_data['Built-in Sort (ms)'].append(builtin_time)
            performance_data['Bubble Sort (ms)'].append(bubble_time)
            
            progress_bar.progress((i + 1) / len(sizes))
        
        status_text.text('Performance tests completed!')
        
        # Display results
        perf_df = pd.DataFrame(performance_data)
        
        # Keep numeric values for plotting
        plot_df = perf_df.copy()
        
        # Format display table with commas and decimal places
        display_df = perf_df.copy()
        display_df['Array Size'] = display_df['Array Size'].apply(lambda x: f'{x:,}')
        
        # Format time values with 2 decimal places
        time_columns = [col for col in display_df.columns if col != 'Array Size' and '(ms)' in col or col.endswith('T)')]
        for col in time_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}' if x is not None else 'N/A')
        
        st.table(display_df)
        
        # Plot performance comparison using numeric data
        fig = go.Figure()
        
        # Plot all algorithm columns except Array Size
        colors = px.colors.qualitative.Plotly
        color_idx = 0
        
        for col in plot_df.columns:
            if col != 'Array Size' and col in plot_df:
                # Handle bubble sort separately (only for small sizes)
                if col == 'Bubble Sort (ms)':
                    bubble_data = plot_df[plot_df[col].notna()]
                    if not bubble_data.empty:
                        fig.add_trace(go.Scatter(
                            x=bubble_data['Array Size'],
                            y=bubble_data[col],
                            mode='lines+markers',
                            name=col.replace(' (ms)', ''),
                            line=dict(color='red')
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=plot_df['Array Size'],
                        y=plot_df[col],
                        mode='lines+markers',
                        name=col.replace(' (ms)', ''),
                        line=dict(color=colors[color_idx % len(colors)])
                    ))
                    color_idx += 1
        
        fig.update_layout(
            title='Sorting Algorithm Performance Comparison',
            xaxis_title='Array Size',
            yaxis_title='Execution Time (ms)',
            yaxis_type='log'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Thread Scalability Analysis
    if use_parallel or test_parallel:
        st.header("Thread Scalability Analysis")
        
        if st.button("Run Thread Scalability Test"):
            st.info("Testing thread scalability with different array sizes...")
            
            test_sizes = [10000, 100000, 1000000]
            test_threads = [1, 2, 4, 8, max_threads, max_threads * 2]
            
            scalability_data = {
                'Array Size': [],
                'Threads': [],
                'Time (ms)': [],
                'Speedup': []
            }
            
            for size in test_sizes:
                test_array = np.random.randint(1, 1000, size).tolist()
                baseline_time = None
                
                for threads in test_threads:
                    if threads > max_threads * 2:
                        continue
                    
                    # Test parallel algorithm
                    if algorithm_type == "Original Reference-Based":
                        sorter = ParallelReferenceBasedSorting(threads)
                    elif algorithm_type == "Optimized Hash-Based":
                        sorter = OptimizedParallelReferenceBasedSorting(threads)
                    else:
                        sorter = ParallelStandardSorting(threads)
                    
                    start_time = time.time()
                    if algorithm_type == "Standard Algorithms":
                        sorter.parallel_merge_sort(test_array.copy())
                    else:
                        sorter.sort(test_array.copy())
                    elapsed_time = (time.time() - start_time) * 1000
                    
                    if baseline_time is None:
                        baseline_time = elapsed_time
                    
                    speedup = baseline_time / elapsed_time
                    
                    scalability_data['Array Size'].append(size)
                    scalability_data['Threads'].append(threads)
                    scalability_data['Time (ms)'].append(elapsed_time)
                    scalability_data['Speedup'].append(speedup)
            
            # Create scalability plot
            scalability_df = pd.DataFrame(scalability_data)
            
            fig = px.line(scalability_df, 
                         x='Threads', 
                         y='Speedup', 
                         color='Array Size',
                         title='Thread Scalability Analysis',
                         markers=True)
            
            # Add ideal speedup line
            ideal_threads = list(range(1, max(test_threads) + 1))
            fig.add_trace(go.Scatter(
                x=ideal_threads,
                y=ideal_threads,
                mode='lines',
                name='Ideal Speedup',
                line=dict(dash='dash', color='gray')
            ))
            
            fig.update_xaxis(title='Number of Threads')
            fig.update_yaxis(title='Speedup Factor')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results
            st.subheader("Detailed Scalability Results")
            display_df = scalability_df.copy()
            display_df['Time (ms)'] = display_df['Time (ms)'].apply(lambda x: f'{x:.2f}')
            display_df['Speedup'] = display_df['Speedup'].apply(lambda x: f'{x:.2f}x')
            st.dataframe(display_df)
    
    # Analysis and insights
    st.header("Analysis and Insights")
    
    # Algorithm optimization suggestions
    st.subheader("Algorithm Optimization Strategies")
    
    with st.expander("Current Algorithm Analysis & Optimization Ideas"):
        st.markdown("""
        **Current Algorithm Performance Issues:**
        
        1. **Space Complexity Problem**: The algorithm creates an array of size `(max_value - min_value + 1)`, which can be enormous for sparse data
        2. **Memory Waste**: Many null positions when values are spread far apart
        3. **No Early Termination**: Always processes the entire range even when unnecessary
        
        **Optimization Strategies:**
        
        **1. Hash-Based Approach (Counting Sort Variant)**
        - Use a hash map/dictionary instead of a positioned array
        - Only store actual values, eliminating null positions
        - Time: O(n), Space: O(k) where k = unique values
        
        **2. Bucket Sort Hybrid**
        - Divide the range into smaller buckets
        - Use efficient sorting within each bucket
        - Reduces memory usage for large ranges
        
        **3. Radix Sort Integration**
        - For large integer ranges, use radix sort principles
        - Process digits from least to most significant
        - Maintains O(n) time complexity
        
        **4. Adaptive Range Detection**
        - Pre-scan to find actual min/max values
        - Use offset indexing to minimize array size
        - Skip unnecessary null positions
        
        **5. In-Place Optimizations**
        - Use bit manipulation for small ranges
        - Implement cyclic sort for consecutive integers
        - Reduce memory allocation overhead
        
        **6. Parallel Processing**
        - Split array into chunks for difference calculation
        - Parallel positioning for independent elements
        - Merge results efficiently
        """)
        
        if st.button("Show Optimized Implementation Demo"):
            st.markdown("**Optimized Hash-Based Version:**")
            st.code('''
def optimized_reference_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Use first element as reference
    reference = arr[0]
    
    # Use dictionary for counting (similar to counting sort)
    count_map = {}
    for num in arr:
        diff = num - reference
        count_map[diff] = count_map.get(diff, 0) + 1
    
    # Build result by iterating through sorted differences
    result = []
    for diff in sorted(count_map.keys()):
        value = diff + reference
        result.extend([value] * count_map[diff])
    
    return result
            ''', language='python')
            
            st.markdown("**Benefits of Optimized Version:**")
            st.markdown("""
            - **Space Efficient**: O(k) space where k = unique values
            - **No Null Positions**: Only stores actual values
            - **Handles Duplicates**: Properly counts repeated elements
            - **Large Range Friendly**: Works well even with sparse data
            - **Maintains Stability**: Preserves relative order of equal elements
            """)

    # Performance characteristics comparison
    st.subheader("Performance Characteristics Comparison")
    
    perf_comparison = pd.DataFrame({
        'Algorithm': ['Reference-Based', 'Python Built-in', 'Quick Sort', 'Merge Sort'],
        'Time Complexity': ['O(n + k)', 'O(n log n)', 'O(n log n)', 'O(n log n)'],
        'Space Complexity': ['O(k)', 'O(1)', 'O(log n)', 'O(n)'],
        'Best Use Case': [
            'Small range, dense data',
            'General purpose',
            'General purpose',
            'Stable sorting needed'
        ],
        'Stability': ['Stable', 'Stable', 'Unstable', 'Stable']
    })
    
    st.table(perf_comparison)
    st.caption("k = range of values (max - min + 1), n = number of elements")
    
    with st.expander("Algorithm Advantages and Disadvantages"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Advantages:**")
            st.markdown("""
            - Simple to understand and implement
            - Good for arrays with small range of values
            - Stable sorting (maintains relative order of equal elements)
            - Linear time complexity in best case
            """)
        
        with col2:
            st.markdown("**Disadvantages:**")
            st.markdown("""
            - Space complexity depends on the range of values
            - Performance degrades with large value ranges
            - Not suitable for negative numbers without modification
            - Memory intensive for sparse data
            """)
    
    # Edge cases testing
    st.header("Edge Cases Testing")
    
    edge_cases = {
        "Empty Array": [],
        "Single Element": [42],
        "All Same Elements": [5, 5, 5, 5],
        "Two Elements": [3, 1],
        "Negative Numbers": [-5, -2, -8, -1],
        "Mixed Positive/Negative": [-2, 5, -1, 3, 0]
    }
    
    if st.button("Test Edge Cases"):
        for case_name, test_array in edge_cases.items():
            st.subheader(f"Testing: {case_name}")
            st.write(f"Input: {test_array}")
            
            try:
                if len(test_array) == 0:
                    st.warning("Empty array - no sorting needed")
                    continue
                
                # Use appropriate sorting method based on algorithm type
                if algorithm_type == "Standard Algorithms":
                    # For standard algorithms, use merge sort as example
                    if use_parallel:
                        result = std_sorter.parallel_merge_sort(test_array.copy())
                    else:
                        result = std_sorter.merge_sort(test_array.copy())
                else:
                    result = ref_sorter.sort(test_array.copy())
                
                st.success(f"Result: {result}")
                
                # Verify correctness
                expected = sorted(test_array)
                if result == expected:
                    st.success("✓ Correct result")
                else:
                    st.error(f"✗ Incorrect result. Expected: {expected}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
