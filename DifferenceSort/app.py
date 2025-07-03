import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
from sorting_algorithms import ReferenceBasedSorting, StandardSorting, OptimizedReferenceBasedSorting
from visualization import SortingVisualizer

def main():
    st.title("Reference-Based Sorting Algorithm Tester")
    st.markdown("### Test and visualize a custom reference-based sorting algorithm")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
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
        algorithm_choice = st.radio(
            "Choose algorithm version to view:",
            ["Original Algorithm", "Optimized Algorithm"],
            horizontal=True
        )
        
        if algorithm_choice == "Original Algorithm":
            st.markdown("""
            **Original Reference-Based Sorting Algorithm:**
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
        else:
            st.markdown("""
            **Optimized Reference-Based Sorting Algorithm:**
            1. **Reference Selection**: The first element becomes the reference point
            2. **Difference Calculation**: Calculate the difference of each element relative to the reference
            3. **Hash Mapping**: Store differences and their values in a hash map (no null positions)
            4. **Sorted Assembly**: Sort differences and build final array from hash map
            
            **Example**: For array [5, 2, 8, 1, 9]
            - Reference: 5
            - Differences: [0, -3, 3, -4, 4]
            - Hash map: {-4: [1], -3: [2], 0: [5], 3: [8], 4: [9]}
            - Final sorted: [1, 2, 5, 8, 9]
            
            **Key Advantages:**
            - No memory waste on null positions
            - Efficient for sparse data
            - Handles duplicates naturally
            """)
        
    # Algorithm selector (outside expander to persist selection)
    st.subheader("Choose Algorithm to Visualize")
    
    # Initialize session state for algorithm selection
    if 'selected_algorithm' not in st.session_state:
        st.session_state.selected_algorithm = "Original Reference-Based"
    
    selected_algorithm = st.selectbox(
        "Select which algorithm to run step-by-step:",
        ["Original Reference-Based", "Optimized Reference-Based"],
        index=0 if st.session_state.selected_algorithm == "Original Reference-Based" else 1,
        key="algorithm_selector"
    )
    
    # Update session state
    st.session_state.selected_algorithm = selected_algorithm
    
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
    
    # Initialize sorting classes
    ref_sorter = ReferenceBasedSorting()
    optimized_sorter = OptimizedReferenceBasedSorting()
    std_sorter = StandardSorting()
    visualizer = SortingVisualizer()
    
    # Execute selected sorting algorithm with steps
    if selected_algorithm == "Original Reference-Based":
        st.header("Original Reference-Based Sorting Steps")
        try:
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
            st.error(f"Error in original reference-based sorting: {str(e)}")
            return
    
    else:  # Optimized Reference-Based
        st.header("Optimized Reference-Based Sorting Steps")
        try:
            steps = optimized_sorter.sort_with_steps(input_array.copy())
            
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
                
                elif step['type'] == 'hash_mapping':
                    st.write("**Hash Map Creation:**")
                    # Display hash map
                    hash_data = []
                    for diff, info in sorted(step['count_map'].items()):
                        hash_data.append({
                            'Difference': diff,
                            'Values': info['values'],
                            'Count': info['count']
                        })
                    hash_df = pd.DataFrame(hash_data)
                    st.table(hash_df)
                    
                    # Visualize hash mapping
                    figures = visualizer.visualize_step_by_step([step])
                    if figures:
                        st.plotly_chart(figures[0], use_container_width=True)
                
                elif step['type'] == 'final_optimized':
                    st.write(f"**Sorted Differences**: {step['sorted_differences']}")
                    st.write(f"**Final Sorted Array**: {step['sorted_array']}")
                    
                    # Visualize final result
                    figures = visualizer.visualize_step_by_step([step])
                    if figures:
                        st.plotly_chart(figures[0], use_container_width=True)
                    
                    # Show before and after
                    comparison_df = pd.DataFrame({
                        'Original': input_array + [None] * (len(step['sorted_array']) - len(input_array)),
                        'Sorted': step['sorted_array'] + [None] * (len(input_array) - len(step['sorted_array']))
                    })
                    st.table(comparison_df)
        
        except Exception as e:
            st.error(f"Error in optimized reference-based sorting: {str(e)}")
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
    
    # Test with different array sizes
    sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    if st.button("Run Performance Tests"):
        st.warning("⚠️ Testing large arrays (up to 10M elements) may take several minutes to complete.")
        
        performance_data = {
            'Array Size': [],
            'Reference-Based (ms)': [],
            'Optimized Reference-Based (ms)': [],
            'Built-in Sort (ms)': [],
            'Bubble Sort (ms)': []
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, size in enumerate(sizes):
            status_text.text(f'Testing array size: {size:,} ({i+1}/{len(sizes)} - {((i+1)/len(sizes)*100):.0f}%)')
            
            # Generate test array
            test_array = np.random.randint(1, 1000, size).tolist()
            
            # Test reference-based sorting
            start_time = time.time()
            ref_sorter.sort(test_array.copy())
            ref_time = (time.time() - start_time) * 1000
            
            # Test optimized reference-based sorting
            start_time = time.time()
            optimized_sorter.sort(test_array.copy())
            optimized_time = (time.time() - start_time) * 1000
            
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
            performance_data['Reference-Based (ms)'].append(ref_time)
            performance_data['Optimized Reference-Based (ms)'].append(optimized_time)
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
        time_columns = ['Reference-Based (ms)', 'Optimized Reference-Based (ms)', 'Built-in Sort (ms)', 'Bubble Sort (ms)']
        for col in time_columns:
            display_df[col] = display_df[col].apply(lambda x: f'{x:.2f}' if x is not None else 'N/A')
        
        st.table(display_df)
        
        # Plot performance comparison using numeric data
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=plot_df['Array Size'],
            y=plot_df['Reference-Based (ms)'],
            mode='lines+markers',
            name='Reference-Based',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=plot_df['Array Size'],
            y=plot_df['Optimized Reference-Based (ms)'],
            mode='lines+markers',
            name='Optimized Reference-Based',
            line=dict(color='purple')
        ))
        
        fig.add_trace(go.Scatter(
            x=plot_df['Array Size'],
            y=plot_df['Built-in Sort (ms)'],
            mode='lines+markers',
            name='Built-in Sort',
            line=dict(color='green')
        ))
        
        # Add bubble sort only for sizes where it was tested
        bubble_data = plot_df[plot_df['Bubble Sort (ms)'].notna()]
        if not bubble_data.empty:
            fig.add_trace(go.Scatter(
                x=bubble_data['Array Size'],
                y=bubble_data['Bubble Sort (ms)'],
                mode='lines+markers',
                name='Bubble Sort',
                line=dict(color='red')
            ))
        
        fig.update_layout(
            title='Sorting Algorithm Performance Comparison',
            xaxis_title='Array Size',
            yaxis_title='Execution Time (ms)',
            yaxis_type='log'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Space efficiency comparison
    st.subheader("Space Efficiency Comparison")
    
    if st.button("Test Space Efficiency"):
        st.markdown("**Testing space efficiency with sparse data:**")
        
        # Test with sparse data (large range, few elements)
        sparse_test_cases = [
            ("Small sparse: [1, 1000]", [1, 1000]),
            ("Medium sparse: [5, 500, 2000]", [5, 500, 2000]),
            ("Large sparse: [1, 10000, 50000]", [1, 10000, 50000]),
            ("Very sparse: [1, 100000]", [1, 100000])
        ]
        
        efficiency_data = []
        
        for case_name, test_array in sparse_test_cases:
            # Get memory usage comparison
            sorted_result, original_space, optimized_space = optimized_sorter.sort_with_memory_usage(test_array.copy())
            
            efficiency_data.append({
                'Test Case': case_name,
                'Array Size': len(test_array),
                'Value Range': f"{min(test_array)} to {max(test_array)}",
                'Original Space Needed': original_space,
                'Optimized Space Used': optimized_space,
                'Space Reduction': f"{((original_space - optimized_space) / original_space * 100):.1f}%"
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        st.table(efficiency_df)
        
        # Visualize space efficiency
        fig_space = go.Figure()
        
        fig_space.add_trace(go.Bar(
            x=efficiency_df['Test Case'],
            y=efficiency_df['Original Space Needed'],
            name='Original Algorithm',
            marker_color='lightcoral'
        ))
        
        fig_space.add_trace(go.Bar(
            x=efficiency_df['Test Case'],
            y=efficiency_df['Optimized Space Used'],
            name='Optimized Algorithm',
            marker_color='lightgreen'
        ))
        
        fig_space.update_layout(
            title='Space Usage Comparison: Original vs Optimized',
            xaxis_title='Test Case',
            yaxis_title='Memory Units Required',
            yaxis_type='log',
            barmode='group'
        )
        
        st.plotly_chart(fig_space, use_container_width=True)
        
        st.success("**Key Insights:**")
        st.write("- Optimized algorithm uses significantly less memory for sparse data")
        st.write("- Memory savings increase dramatically with larger value ranges")
        st.write("- Original algorithm struggles with large gaps between values")
        st.write("- Optimized version maintains performance regardless of value distribution")
    
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
        'Algorithm': ['Original Reference-Based', 'Optimized Hash-Based', 'Python Built-in', 'Quick Sort', 'Merge Sort'],
        'Time Complexity': ['O(n + k)', 'O(n log k)', 'O(n log n)', 'O(n log n)', 'O(n log n)'],
        'Space Complexity': ['O(k)', 'O(k)', 'O(1)', 'O(log n)', 'O(n)'],
        'Best Use Case': [
            'Small range, dense data',
            'Any range, sparse data',
            'General purpose',
            'General purpose',
            'Stable sorting needed'
        ],
        'Stability': ['Stable', 'Stable', 'Stable', 'Unstable', 'Stable']
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
