import time
import math
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
import multiprocessing as mp
import numpy as np

class MultiprocessReferenceBasedSorting:
    """
    Multiprocess implementation of the reference-based sorting algorithm.
    Uses true parallelism via multiprocessing instead of threading.
    """
    
    def __init__(self, num_processes: Optional[int] = None):
        """
        Initialize with specified number of processes.
        
        Args:
            num_processes: Number of processes to use. If None, uses CPU count.
        """
        self.num_processes = num_processes or cpu_count()
        
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort the array using multiprocess reference-based algorithm.
        """
        if not arr or len(arr) <= 1:
            return arr
        
        # For small arrays, use sequential version
        if len(arr) < 10000:
            return self._sequential_sort(arr)
        
        # Step 1: Select reference (first element)
        reference = arr[0]
        
        # Step 2: Calculate differences in parallel
        differences = self._parallel_calculate_differences(arr, reference)
        
        # Step 3: Find the range for positioning
        min_diff = min(differences)
        max_diff = max(differences)
        
        # Step 4: Position elements using parallel counting
        count_map = self._parallel_count_differences(arr, reference, differences)
        
        # Step 5: Build result
        result = []
        for diff in sorted(count_map.keys()):
            value = diff + reference
            result.extend([value] * count_map[diff])
        
        return result
    
    def _sequential_sort(self, arr: List[int]) -> List[int]:
        """Sequential version for small arrays."""
        if not arr:
            return []
        
        reference = arr[0]
        count_map = {}
        
        for num in arr:
            diff = num - reference
            count_map[diff] = count_map.get(diff, 0) + 1
        
        result = []
        for diff in sorted(count_map.keys()):
            value = diff + reference
            result.extend([value] * count_map[diff])
        
        return result
    
    def _parallel_calculate_differences(self, arr: List[int], reference: int) -> List[int]:
        """Calculate differences in parallel using multiprocessing."""
        chunk_size = math.ceil(len(arr) / self.num_processes)
        chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]
        
        with Pool(processes=self.num_processes) as pool:
            # Use starmap to pass multiple arguments
            chunk_diffs = pool.starmap(
                _calculate_differences_chunk,
                [(chunk, reference) for chunk in chunks]
            )
        
        # Flatten the results
        differences = []
        for chunk_diff in chunk_diffs:
            differences.extend(chunk_diff)
        
        return differences
    
    def _parallel_count_differences(self, arr: List[int], reference: int, 
                                   differences: List[int]) -> Dict[int, int]:
        """Count differences in parallel using multiprocessing."""
        chunk_size = math.ceil(len(arr) / self.num_processes)
        chunks = [(arr[i:i + chunk_size], reference) 
                  for i in range(0, len(arr), chunk_size)]
        
        with Pool(processes=self.num_processes) as pool:
            local_counts = pool.starmap(_count_differences_chunk, chunks)
        
        # Merge all local counts
        global_count = {}
        for local_count in local_counts:
            for diff, count in local_count.items():
                global_count[diff] = global_count.get(diff, 0) + count
        
        return global_count
    
    def sort_with_steps(self, arr: List[int]) -> List[Dict[str, Any]]:
        """Sort with step tracking for visualization."""
        steps = []
        
        if not arr:
            return [{'type': 'final', 'description': 'Empty array', 'sorted_array': []}]
        
        if len(arr) == 1:
            return [{'type': 'final', 'description': 'Single element array', 'sorted_array': arr}]
        
        # Step 1: Reference selection
        reference = arr[0]
        steps.append({
            'type': 'reference',
            'description': f'Select reference element - Using {self.num_processes} processes',
            'reference': reference,
            'array': arr.copy(),
            'num_processes': self.num_processes
        })
        
        # Step 2: Calculate differences
        start_time = time.time()
        if len(arr) < 10000:
            differences = [x - reference for x in arr]
        else:
            differences = self._parallel_calculate_differences(arr, reference)
        diff_time = time.time() - start_time
        
        steps.append({
            'type': 'differences',
            'description': f'Calculate differences ({"parallel" if len(arr) >= 10000 else "sequential"}, {diff_time*1000:.2f}ms)',
            'reference': reference,
            'array': arr.copy(),
            'differences': differences,
            'parallel_time': diff_time
        })
        
        # Step 3: Count and build result
        if len(arr) < 10000:
            result = self._sequential_sort(arr)
        else:
            result = self.sort(arr)
        
        steps.append({
            'type': 'final',
            'description': 'Final sorted array',
            'sorted_array': result
        })
        
        return steps


class MultiprocessStandardSorting:
    """
    Multiprocess implementations of standard sorting algorithms.
    """
    
    def __init__(self, num_processes: Optional[int] = None):
        """Initialize with specified number of processes."""
        self.num_processes = num_processes or cpu_count()
    
    def parallel_merge_sort(self, arr: List[int]) -> List[int]:
        """Multiprocess merge sort implementation."""
        if len(arr) <= 1:
            return arr
        
        # Base case for parallelism
        if len(arr) < 10000 or self.num_processes == 1:
            return self._sequential_merge_sort(arr)
        
        # Split array
        mid = len(arr) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]
        
        # Use multiprocessing for large arrays
        with Pool(processes=2) as pool:
            results = pool.map(self._sequential_merge_sort, [left_arr, right_arr])
        
        return self._merge(results[0], results[1])
    
    def parallel_quick_sort(self, arr: List[int]) -> List[int]:
        """Multiprocess quick sort with work stealing."""
        if len(arr) <= 1:
            return arr
        
        if len(arr) < 10000 or self.num_processes == 1:
            return self._sequential_quick_sort(arr)
        
        # Three-way partitioning for better parallelism
        pivot = arr[len(arr) // 2]
        less = []
        equal = []
        greater = []
        
        for x in arr:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            else:
                greater.append(x)
        
        # Parallel sort of partitions
        with Pool(processes=2) as pool:
            results = pool.map(self._sequential_quick_sort, [less, greater])
        
        return results[0] + equal + results[1]
    
    def parallel_bucket_sort(self, arr: List[int]) -> List[int]:
        """
        Multiprocess bucket sort - efficient for uniformly distributed data.
        """
        if len(arr) <= 1:
            return arr
        
        if len(arr) < 10000:
            return sorted(arr)
        
        # Find range
        min_val = min(arr)
        max_val = max(arr)
        
        if min_val == max_val:
            return arr
        
        # Create buckets
        num_buckets = min(self.num_processes, len(arr) // 1000)
        bucket_range = (max_val - min_val) / num_buckets
        buckets = [[] for _ in range(num_buckets)]
        
        # Distribute elements into buckets
        for num in arr:
            index = min(int((num - min_val) / bucket_range), num_buckets - 1)
            buckets[index].append(num)
        
        # Sort buckets in parallel
        with Pool(processes=self.num_processes) as pool:
            sorted_buckets = pool.map(sorted, buckets)
        
        # Concatenate results
        result = []
        for bucket in sorted_buckets:
            result.extend(bucket)
        
        return result
    
    def _sequential_merge_sort(self, arr: List[int]) -> List[int]:
        """Sequential merge sort for smaller arrays."""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self._sequential_merge_sort(arr[:mid])
        right = self._sequential_merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _sequential_quick_sort(self, arr: List[int]) -> List[int]:
        """Sequential quick sort for smaller arrays."""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return (self._sequential_quick_sort(left) + 
                middle + 
                self._sequential_quick_sort(right))
    
    def _merge(self, left: List[int], right: List[int]) -> List[int]:
        """Merge two sorted arrays."""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result


# Helper functions for multiprocessing (must be at module level)
def _calculate_differences_chunk(chunk: List[int], reference: int) -> List[int]:
    """Calculate differences for a chunk of the array."""
    return [x - reference for x in chunk]


def _count_differences_chunk(chunk_ref: Tuple[List[int], int]) -> Dict[int, int]:
    """Count differences in a chunk."""
    chunk, reference = chunk_ref
    count_map = {}
    
    for num in chunk:
        diff = num - reference
        count_map[diff] = count_map.get(diff, 0) + 1
    
    return count_map