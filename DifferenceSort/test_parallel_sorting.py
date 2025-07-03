"""
Test script for parallel sorting algorithms with different thread counts.
"""

import time
import random
import multiprocessing
from sorting_algorithms import ReferenceBasedSorting, OptimizedReferenceBasedSorting
from parallel_sorting_algorithms import (
    ParallelReferenceBasedSorting,
    ParallelStandardSorting,
    OptimizedParallelReferenceBasedSorting
)


def test_correctness():
    """Test that all algorithms produce correct results."""
    print("Testing correctness of parallel algorithms...")
    
    # Test data
    test_arrays = [
        [],  # Empty
        [42],  # Single element
        [3, 1],  # Two elements
        [5, 2, 8, 1, 9],  # Small array
        list(range(100, 0, -1)),  # Reverse sorted
        [1] * 50,  # All same
        random.sample(range(1000), 100)  # Random
    ]
    
    algorithms = [
        ("Sequential Reference-Based", ReferenceBasedSorting()),
        ("Parallel Reference-Based (2T)", ParallelReferenceBasedSorting(2)),
        ("Parallel Reference-Based (4T)", ParallelReferenceBasedSorting(4)),
        ("Optimized Sequential", OptimizedReferenceBasedSorting()),
        ("Optimized Parallel (2T)", OptimizedParallelReferenceBasedSorting(2)),
        ("Optimized Parallel (4T)", OptimizedParallelReferenceBasedSorting(4)),
    ]
    
    for test_array in test_arrays:
        expected = sorted(test_array)
        print(f"\nTesting array of size {len(test_array)}: {test_array[:5]}...")
        
        for name, algorithm in algorithms:
            try:
                result = algorithm.sort(test_array.copy())
                if result == expected:
                    print(f"  ✓ {name}: PASS")
                else:
                    print(f"  ✗ {name}: FAIL - Got {result[:5]}...")
            except Exception as e:
                print(f"  ✗ {name}: ERROR - {str(e)}")


def test_performance():
    """Test performance with different thread counts."""
    print("\nTesting performance with different thread counts...")
    
    sizes = [10000, 100000, 1000000]
    thread_counts = [1, 2, 4, 8, multiprocessing.cpu_count()]
    
    for size in sizes:
        print(f"\nArray size: {size:,}")
        test_array = [random.randint(1, 1000) for _ in range(size)]
        
        # Test Reference-Based
        print("  Reference-Based Sorting:")
        baseline_time = None
        
        for threads in thread_counts:
            if threads == 1:
                sorter = ReferenceBasedSorting()
            else:
                sorter = ParallelReferenceBasedSorting(threads)
            
            start = time.time()
            sorter.sort(test_array.copy())
            elapsed = time.time() - start
            
            if baseline_time is None:
                baseline_time = elapsed
            
            speedup = baseline_time / elapsed
            print(f"    {threads} threads: {elapsed:.3f}s (speedup: {speedup:.2f}x)")
        
        # Test Optimized Hash-Based
        print("  Optimized Hash-Based Sorting:")
        baseline_time = None
        
        for threads in thread_counts:
            if threads == 1:
                sorter = OptimizedReferenceBasedSorting()
            else:
                sorter = OptimizedParallelReferenceBasedSorting(threads)
            
            start = time.time()
            sorter.sort(test_array.copy())
            elapsed = time.time() - start
            
            if baseline_time is None:
                baseline_time = elapsed
            
            speedup = baseline_time / elapsed
            print(f"    {threads} threads: {elapsed:.3f}s (speedup: {speedup:.2f}x)")


def test_parallel_standard_sorts():
    """Test parallel versions of standard sorting algorithms."""
    print("\nTesting parallel standard sorting algorithms...")
    
    size = 100000
    test_array = [random.randint(1, 1000) for _ in range(size)]
    
    sorter_seq = ParallelStandardSorting(1)
    sorter_par = ParallelStandardSorting(4)
    
    # Test Merge Sort
    print(f"\nMerge Sort (array size: {size:,}):")
    
    start = time.time()
    result1 = sorter_seq.parallel_merge_sort(test_array.copy())
    time1 = time.time() - start
    
    start = time.time()
    result2 = sorter_par.parallel_merge_sort(test_array.copy())
    time2 = time.time() - start
    
    print(f"  Sequential: {time1:.3f}s")
    print(f"  Parallel (4T): {time2:.3f}s (speedup: {time1/time2:.2f}x)")
    print(f"  Results match: {result1 == result2}")
    
    # Test Quick Sort
    print(f"\nQuick Sort (array size: {size:,}):")
    
    start = time.time()
    result1 = sorter_seq.parallel_quick_sort(test_array.copy())
    time1 = time.time() - start
    
    start = time.time()
    result2 = sorter_par.parallel_quick_sort(test_array.copy())
    time2 = time.time() - start
    
    print(f"  Sequential: {time1:.3f}s")
    print(f"  Parallel (4T): {time2:.3f}s (speedup: {time1/time2:.2f}x)")
    print(f"  Results match: {result1 == result2}")
    
    # Test Sample Sort
    print(f"\nSample Sort (array size: {size:,}):")
    
    start = time.time()
    result1 = sorter_seq.parallel_sample_sort(test_array.copy())
    time1 = time.time() - start
    
    start = time.time()
    result2 = sorter_par.parallel_sample_sort(test_array.copy())
    time2 = time.time() - start
    
    print(f"  Sequential: {time1:.3f}s")
    print(f"  Parallel (4T): {time2:.3f}s (speedup: {time1/time2:.2f}x)")
    print(f"  Results match: {result1 == result2}")


if __name__ == "__main__":
    print(f"CPU cores available: {multiprocessing.cpu_count()}")
    print("=" * 60)
    
    test_correctness()
    print("\n" + "=" * 60)
    
    test_performance()
    print("\n" + "=" * 60)
    
    test_parallel_standard_sorts()
    print("\n" + "=" * 60)
    print("\nAll tests completed!")