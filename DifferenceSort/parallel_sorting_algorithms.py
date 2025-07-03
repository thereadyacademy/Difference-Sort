import time
import math
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing

class ParallelReferenceBasedSorting:
    """
    Parallel implementation of the custom reference-based sorting algorithm.
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        """
        Initialize parallel sorting with specified number of threads.
        
        Args:
            num_threads: Number of threads to use. If None, uses CPU count.
        """
        self.num_threads = num_threads or multiprocessing.cpu_count()
        
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort the array using parallel reference-based algorithm.
        
        Args:
            arr: List of integers to sort
            
        Returns:
            Sorted list of integers
        """
        if not arr or len(arr) <= 1:
            return arr
        
        # Step 1: Select reference (first element)
        reference = arr[0]
        
        # Step 2: Calculate differences in parallel
        differences = self._parallel_calculate_differences(arr, reference)
        
        # Step 3: Find the range for positioning
        min_diff = min(differences)
        max_diff = max(differences)
        
        # Step 4: Position elements in parallel
        positioned_array = self._parallel_position_elements(arr, differences, min_diff, max_diff)
        
        # Step 5: Flatten and return result
        result = []
        for item in positioned_array:
            if item:  # if list is not empty
                item.sort()  # Sort elements at same position
                result.extend(item)
        
        return result
    
    def sort_with_steps(self, arr: List[int]) -> List[Dict[str, Any]]:
        """
        Sort the array and return detailed steps for visualization.
        """
        steps = []
        
        if not arr:
            return [{'type': 'final', 'description': 'Empty array', 'sorted_array': []}]
        
        if len(arr) == 1:
            return [{'type': 'final', 'description': 'Single element array', 'sorted_array': arr}]
        
        # Step 1: Reference selection
        reference = arr[0]
        steps.append({
            'type': 'reference',
            'description': f'Select reference element (first element) - Using {self.num_threads} threads',
            'reference': reference,
            'array': arr.copy(),
            'num_threads': self.num_threads
        })
        
        # Step 2: Calculate differences in parallel
        start_time = time.time()
        differences = self._parallel_calculate_differences(arr, reference)
        diff_time = time.time() - start_time
        
        steps.append({
            'type': 'differences',
            'description': f'Calculate differences in parallel ({diff_time*1000:.2f}ms)',
            'reference': reference,
            'array': arr.copy(),
            'differences': differences,
            'parallel_time': diff_time
        })
        
        # Step 3: Positioning
        min_diff = min(differences)
        max_diff = max(differences)
        
        start_time = time.time()
        positioned_array = self._parallel_position_elements(arr, differences, min_diff, max_diff)
        pos_time = time.time() - start_time
        
        # Convert to display format
        display_positioned = []
        for item in positioned_array:
            if not item:
                display_positioned.append(None)
            else:
                display_positioned.extend(item)
        
        steps.append({
            'type': 'positioning',
            'description': f'Position elements in parallel ({pos_time*1000:.2f}ms)',
            'reference': reference,
            'differences': differences,
            'positioned_array': display_positioned,
            'min_diff': min_diff,
            'max_diff': max_diff,
            'parallel_time': pos_time
        })
        
        # Step 4: Final result
        result = []
        for item in positioned_array:
            if item:
                item.sort()
                result.extend(item)
        
        steps.append({
            'type': 'final',
            'description': 'Final sorted array',
            'sorted_array': result
        })
        
        return steps
    
    def _parallel_calculate_differences(self, arr: List[int], reference: int) -> List[int]:
        """Calculate differences in parallel."""
        chunk_size = max(1, len(arr) // self.num_threads)
        differences = [0] * len(arr)
        
        def calculate_chunk(start_idx: int, end_idx: int):
            for i in range(start_idx, end_idx):
                differences[i] = arr[i] - reference
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(0, len(arr), chunk_size):
                end_idx = min(i + chunk_size, len(arr))
                futures.append(executor.submit(calculate_chunk, i, end_idx))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        return differences
    
    def _parallel_position_elements(self, arr: List[int], differences: List[int], 
                                   min_diff: int, max_diff: int) -> List[List[int]]:
        """Position elements in parallel using thread-safe operations."""
        positioned_size = max_diff - min_diff + 1
        positioned_array = [[] for _ in range(positioned_size)]
        locks = [Lock() for _ in range(positioned_size)]
        
        chunk_size = max(1, len(arr) // self.num_threads)
        
        def position_chunk(start_idx: int, end_idx: int):
            for i in range(start_idx, end_idx):
                pos = differences[i] - min_diff
                with locks[pos]:
                    positioned_array[pos].append(arr[i])
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(0, len(arr), chunk_size):
                end_idx = min(i + chunk_size, len(arr))
                futures.append(executor.submit(position_chunk, i, end_idx))
            
            # Wait for all to complete
            for future in as_completed(futures):
                future.result()
        
        return positioned_array


class ParallelStandardSorting:
    """
    Parallel implementations of standard sorting algorithms.
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        """
        Initialize parallel sorting with specified number of threads.
        
        Args:
            num_threads: Number of threads to use. If None, uses CPU count.
        """
        self.num_threads = num_threads or multiprocessing.cpu_count()
    
    def parallel_merge_sort(self, arr: List[int]) -> List[int]:
        """Parallel implementation of merge sort."""
        if len(arr) <= 1:
            return arr
        
        # Base case for parallelism - switch to sequential for small arrays
        if len(arr) < 1000:
            return self._sequential_merge_sort(arr)
        
        mid = len(arr) // 2
        
        # Parallel execution of left and right halves
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_left = executor.submit(self.parallel_merge_sort, arr[:mid])
            future_right = executor.submit(self.parallel_merge_sort, arr[mid:])
            
            left = future_left.result()
            right = future_right.result()
        
        return self._merge(left, right)
    
    def parallel_quick_sort(self, arr: List[int]) -> List[int]:
        """Parallel implementation of quick sort."""
        if len(arr) <= 1:
            return arr
        
        # Base case for parallelism
        if len(arr) < 1000:
            return self._sequential_quick_sort(arr)
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        # Parallel execution of partitions
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_left = executor.submit(self.parallel_quick_sort, left)
            future_right = executor.submit(self.parallel_quick_sort, right)
            
            sorted_left = future_left.result()
            sorted_right = future_right.result()
        
        return sorted_left + middle + sorted_right
    
    def parallel_sample_sort(self, arr: List[int]) -> List[int]:
        """
        Parallel sample sort - better for larger arrays and more threads.
        """
        if len(arr) <= 1:
            return arr
        
        if len(arr) < self.num_threads * 100:
            return sorted(arr)
        
        # Step 1: Sample and find splitters
        sample_size = min(self.num_threads * 10, len(arr))
        samples = sorted([arr[i] for i in range(0, len(arr), len(arr) // sample_size)])
        splitters = [samples[i * len(samples) // self.num_threads] 
                    for i in range(1, self.num_threads)]
        
        # Step 2: Partition data into buckets
        buckets = [[] for _ in range(self.num_threads)]
        
        def assign_to_bucket(elem):
            for i, splitter in enumerate(splitters):
                if elem < splitter:
                    return i
            return self.num_threads - 1
        
        # Parallel partitioning
        chunk_size = len(arr) // self.num_threads
        bucket_locks = [Lock() for _ in range(self.num_threads)]
        
        def partition_chunk(start_idx: int, end_idx: int):
            local_buckets = [[] for _ in range(self.num_threads)]
            for i in range(start_idx, end_idx):
                bucket_idx = assign_to_bucket(arr[i])
                local_buckets[bucket_idx].append(arr[i])
            
            # Merge local buckets to global buckets
            for i, local_bucket in enumerate(local_buckets):
                if local_bucket:
                    with bucket_locks[i]:
                        buckets[i].extend(local_bucket)
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(0, len(arr), chunk_size):
                end_idx = min(i + chunk_size, len(arr))
                futures.append(executor.submit(partition_chunk, i, end_idx))
            
            for future in as_completed(futures):
                future.result()
        
        # Step 3: Sort buckets in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            sorted_buckets = list(executor.map(sorted, buckets))
        
        # Step 4: Concatenate results
        result = []
        for bucket in sorted_buckets:
            result.extend(bucket)
        
        return result
    
    def _sequential_merge_sort(self, arr: List[int]) -> List[int]:
        """Sequential merge sort for small arrays."""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self._sequential_merge_sort(arr[:mid])
        right = self._sequential_merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
    def _sequential_quick_sort(self, arr: List[int]) -> List[int]:
        """Sequential quick sort for small arrays."""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self._sequential_quick_sort(left) + middle + self._sequential_quick_sort(right)
    
    def _merge(self, left: List[int], right: List[int]) -> List[int]:
        """Helper method for merge sort."""
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


class OptimizedParallelReferenceBasedSorting:
    """
    Optimized parallel version using hash-based approach.
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        """
        Initialize parallel sorting with specified number of threads.
        """
        self.num_threads = num_threads or multiprocessing.cpu_count()
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort using optimized parallel hash-based approach.
        """
        if len(arr) <= 1:
            return arr
        
        # Use first element as reference
        reference = arr[0]
        
        # Parallel counting
        count_map = self._parallel_count_differences(arr, reference)
        
        # Build result by iterating through sorted differences
        result = []
        for diff in sorted(count_map.keys()):
            value = diff + reference
            result.extend([value] * count_map[diff])
        
        return result
    
    def _parallel_count_differences(self, arr: List[int], reference: int) -> Dict[int, int]:
        """Count differences in parallel using thread-local dictionaries."""
        chunk_size = max(1, len(arr) // self.num_threads)
        local_maps = []
        lock = Lock()
        
        def count_chunk(start_idx: int, end_idx: int) -> Dict[int, int]:
            local_map = {}
            for i in range(start_idx, end_idx):
                diff = arr[i] - reference
                local_map[diff] = local_map.get(diff, 0) + 1
            return local_map
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(0, len(arr), chunk_size):
                end_idx = min(i + chunk_size, len(arr))
                futures.append(executor.submit(count_chunk, i, end_idx))
            
            # Merge results
            global_map = {}
            for future in as_completed(futures):
                local_map = future.result()
                for diff, count in local_map.items():
                    global_map[diff] = global_map.get(diff, 0) + count
        
        return global_map