import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

class ReferenceBasedSorting:
    """
    Implementation of the custom reference-based sorting algorithm.
    
    Algorithm Steps:
    1. Use the first element as reference
    2. Calculate differences relative to the reference
    3. Position elements based on their differences
    4. Remove null values to get final sorted array
    """
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort the array using reference-based algorithm.
        
        Args:
            arr: List of integers to sort
            
        Returns:
            Sorted list of integers
        """
        if not arr:
            return []
        
        if len(arr) == 1:
            return arr
        
        # Step 1: Select reference (first element)
        reference = arr[0]
        
        # Step 2: Calculate differences
        differences = [x - reference for x in arr]
        
        # Step 3: Find the range for positioning
        min_diff = min(differences)
        max_diff = max(differences)
        
        # Create positioned array with nulls
        positioned_size = max_diff - min_diff + 1
        positioned_array: List[List[int]] = [[] for _ in range(positioned_size)]
        
        # Place elements in their positions
        for i, diff in enumerate(differences):
            pos = diff - min_diff
            positioned_array[pos].append(arr[i])
        
        # Step 4: Remove nulls and flatten
        result = []
        for item in positioned_array:
            if item:  # if list is not empty
                # Sort elements at the same position (for stability)
                item.sort()
                result.extend(item)
        
        return result
    
    def sort_with_steps(self, arr: List[int]) -> List[Dict[str, Any]]:
        """
        Sort the array and return detailed steps for visualization.
        
        Args:
            arr: List of integers to sort
            
        Returns:
            List of dictionaries containing step information
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
            'description': 'Select reference element (first element)',
            'reference': reference,
            'array': arr.copy()
        })
        
        # Step 2: Calculate differences
        differences = [x - reference for x in arr]
        steps.append({
            'type': 'differences',
            'description': 'Calculate differences relative to reference',
            'reference': reference,
            'array': arr.copy(),
            'differences': differences
        })
        
        # Step 3: Positioning
        min_diff = min(differences)
        max_diff = max(differences)
        positioned_size = max_diff - min_diff + 1
        positioned_array: List[List[int]] = [[] for _ in range(positioned_size)]
        
        # Place elements in their positions
        for i, diff in enumerate(differences):
            pos = diff - min_diff
            positioned_array[pos].append(arr[i])
        
        # Convert to display format
        display_positioned = []
        for item in positioned_array:
            if not item:  # if list is empty
                display_positioned.append(None)
            else:
                display_positioned.extend(item)
        
        steps.append({
            'type': 'positioning',
            'description': 'Position elements based on differences',
            'reference': reference,
            'differences': differences,
            'positioned_array': display_positioned,
            'min_diff': min_diff,
            'max_diff': max_diff
        })
        
        # Step 4: Remove nulls and get final result
        result = []
        for item in positioned_array:
            if item:  # if list is not empty
                item.sort()  # Sort elements at same position
                result.extend(item)
        
        steps.append({
            'type': 'final',
            'description': 'Remove null values to get final sorted array',
            'sorted_array': result
        })
        
        return steps

class StandardSorting:
    """
    Implementation of standard sorting algorithms for comparison.
    """
    
    def builtin_sort(self, arr: List[int]) -> List[int]:
        """Use Python's built-in sort (Timsort)."""
        arr.sort()
        return arr
    
    def bubble_sort(self, arr: List[int]) -> List[int]:
        """Implementation of bubble sort."""
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def quick_sort(self, arr: List[int]) -> List[int]:
        """Implementation of quick sort."""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self.quick_sort(left) + middle + self.quick_sort(right)
    
    def merge_sort(self, arr: List[int]) -> List[int]:
        """Implementation of merge sort."""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = self.merge_sort(arr[:mid])
        right = self.merge_sort(arr[mid:])
        
        return self._merge(left, right)
    
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


class OptimizedReferenceBasedSorting:
    """
    Optimized hash-based implementation of the reference-based sorting algorithm.
    Eliminates null positions and handles sparse data efficiently.
    """
    
    def __init__(self, num_threads: Optional[int] = None):
        """
        Initialize with optional thread control.
        
        Args:
            num_threads: Number of threads to use. If None, uses 1 (sequential).
        """
        self.num_threads = num_threads or 1
    
    def sort(self, arr: List[int]) -> List[int]:
        """
        Sort using optimized hash-based approach.
        
        Args:
            arr: List of integers to sort
            
        Returns:
            Sorted list of integers
        """
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
            'description': 'Select reference element (optimized hash-based)',
            'reference': reference,
            'array': arr.copy(),
            'algorithm': 'Optimized Hash-Based'
        })
        
        # Step 2: Calculate differences and count
        count_map = {}
        differences = []
        for num in arr:
            diff = num - reference
            differences.append(diff)
            count_map[diff] = count_map.get(diff, 0) + 1
        
        steps.append({
            'type': 'differences',
            'description': 'Calculate differences and count occurrences',
            'reference': reference,
            'array': arr.copy(),
            'differences': differences,
            'count_map': dict(sorted(count_map.items()))
        })
        
        # Step 3: Build result
        result = []
        for diff in sorted(count_map.keys()):
            value = diff + reference
            result.extend([value] * count_map[diff])
        
        steps.append({
            'type': 'final',
            'description': 'Build sorted array from hash map',
            'sorted_array': result,
            'space_saved': f'{len(count_map)} unique values vs {max(differences) - min(differences) + 1} positions'
        })
        
        return steps
