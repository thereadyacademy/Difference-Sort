import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Optional

class SortingVisualizer:
    """
    Visualization utilities for sorting algorithms.
    """
    
    def visualize_positioning(self, positioned_array: List, reference: int, differences: List[int]) -> go.Figure:
        """
        Visualize the positioning step of the reference-based sorting algorithm.
        
        Args:
            positioned_array: Array with elements positioned and nulls
            reference: Reference element
            differences: List of differences
            
        Returns:
            Plotly figure showing the positioning
        """
        fig = go.Figure()
        
        # Create positions
        positions = list(range(len(positioned_array)))
        
        # Separate non-null and null positions
        non_null_positions = []
        non_null_values = []
        null_positions = []
        
        for i, val in enumerate(positioned_array):
            if val is not None:
                non_null_positions.append(i)
                non_null_values.append(val)
            else:
                null_positions.append(i)
        
        # Plot non-null values
        if non_null_values:
            fig.add_trace(go.Scatter(
                x=non_null_positions,
                y=non_null_values,
                mode='markers+text',
                marker=dict(size=20, color='blue'),
                text=non_null_values,
                textposition="middle center",
                textfont=dict(color='white', size=12),
                name='Elements',
                showlegend=True
            ))
        
        # Plot null positions
        if null_positions:
            fig.add_trace(go.Scatter(
                x=null_positions,
                y=[0] * len(null_positions),
                mode='markers',
                marker=dict(size=15, color='lightgray', symbol='x'),
                name='Null positions',
                showlegend=True
            ))
        
        # Highlight reference element
        if reference in non_null_values:
            ref_pos = non_null_positions[non_null_values.index(reference)]
            fig.add_trace(go.Scatter(
                x=[ref_pos],
                y=[reference],
                mode='markers',
                marker=dict(size=25, color='red', symbol='star'),
                name=f'Reference ({reference})',
                showlegend=True
            ))
        
        fig.update_layout(
            title='Element Positioning Based on Differences',
            xaxis_title='Position Index',
            yaxis_title='Element Value',
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def visualize_array_transformation(self, original: List[int], sorted_array: List[int]) -> go.Figure:
        """
        Visualize the transformation from original to sorted array.
        
        Args:
            original: Original array
            sorted_array: Sorted array
            
        Returns:
            Plotly figure showing before and after
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Array', 'Sorted Array'),
            vertical_spacing=0.3
        )
        
        # Original array
        fig.add_trace(
            go.Bar(
                x=list(range(len(original))),
                y=original,
                name='Original',
                marker_color='lightblue',
                text=original,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Sorted array
        fig.add_trace(
            go.Bar(
                x=list(range(len(sorted_array))),
                y=sorted_array,
                name='Sorted',
                marker_color='lightgreen',
                text=sorted_array,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Array Transformation',
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(title_text="Index", row=1, col=1)
        fig.update_xaxes(title_text="Index", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        return fig
    
    def visualize_performance_comparison(self, performance_data: dict) -> go.Figure:
        """
        Visualize performance comparison between different sorting algorithms.
        
        Args:
            performance_data: Dictionary containing performance metrics
            
        Returns:
            Plotly figure showing performance comparison
        """
        fig = go.Figure()
        
        for algorithm, times in performance_data.items():
            if algorithm != 'Array Size':
                fig.add_trace(go.Scatter(
                    x=performance_data['Array Size'],
                    y=times,
                    mode='lines+markers',
                    name=algorithm,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title='Sorting Algorithm Performance Comparison',
            xaxis_title='Array Size',
            yaxis_title='Execution Time (ms)',
            yaxis_type='log',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def visualize_step_by_step(self, steps: List[dict]) -> List[go.Figure]:
        """
        Create visualizations for each step of the sorting process.
        
        Args:
            steps: List of step dictionaries
            
        Returns:
            List of Plotly figures, one for each step
        """
        figures = []
        
        for step in steps:
            if step['type'] == 'reference':
                fig = self._visualize_reference_step(step)
            elif step['type'] == 'differences':
                fig = self._visualize_differences_step(step)
            elif step['type'] == 'positioning':
                fig = self._visualize_positioning_step(step)
            elif step['type'] == 'final':
                fig = self._visualize_final_step(step)
            elif step['type'] == 'hash_mapping':
                fig = self._visualize_hash_mapping_step(step)
            elif step['type'] == 'final_optimized':
                fig = self._visualize_final_optimized_step(step)
            else:
                continue
            
            figures.append(fig)
        
        return figures
    
    def _visualize_reference_step(self, step: dict) -> go.Figure:
        """Visualize reference selection step."""
        fig = go.Figure()
        
        array = step['array']
        reference = step['reference']
        
        colors = ['red' if x == reference else 'lightblue' for x in array]
        
        fig.add_trace(go.Bar(
            x=list(range(len(array))),
            y=array,
            marker_color=colors,
            text=array,
            textposition='auto',
            name='Array Elements'
        ))
        
        fig.update_layout(
            title=f'Reference Selection: {reference}',
            xaxis_title='Index',
            yaxis_title='Value',
            showlegend=False
        )
        
        return fig
    
    def _visualize_differences_step(self, step: dict) -> go.Figure:
        """Visualize differences calculation step."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Values', 'Differences from Reference'),
            vertical_spacing=0.3
        )
        
        array = step['array']
        differences = step['differences']
        reference = step['reference']
        
        # Original values
        fig.add_trace(
            go.Bar(
                x=list(range(len(array))),
                y=array,
                name='Original',
                marker_color='lightblue',
                text=array,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Differences
        colors = ['red' if d == 0 else 'green' if d > 0 else 'orange' for d in differences]
        fig.add_trace(
            go.Bar(
                x=list(range(len(differences))),
                y=differences,
                name='Differences',
                marker_color=colors,
                text=differences,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Differences from Reference ({reference})',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def _visualize_positioning_step(self, step: dict) -> go.Figure:
        """Visualize positioning step."""
        return self.visualize_positioning(
            step['positioned_array'],
            step['reference'],
            step['differences']
        )
    
    def _visualize_final_step(self, step: dict) -> go.Figure:
        """Visualize final result step."""
        fig = go.Figure()
        
        sorted_array = step['sorted_array']
        
        fig.add_trace(go.Bar(
            x=list(range(len(sorted_array))),
            y=sorted_array,
            marker_color='lightgreen',
            text=sorted_array,
            textposition='auto',
            name='Sorted Array'
        ))
        
        fig.update_layout(
            title='Final Sorted Array',
            xaxis_title='Index',
            yaxis_title='Value',
            showlegend=False
        )
        
        return fig
    
    def _visualize_hash_mapping_step(self, step: dict) -> go.Figure:
        """Visualize hash mapping step for optimized algorithm."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Array Elements', 'Hash Map: Differences → Values'),
            vertical_spacing=0.4,
            specs=[[{"secondary_y": False}], [{"type": "table"}]]
        )
        
        array = step['array']
        reference = step['reference']
        count_map = step['count_map']
        
        # Original array with reference highlighted
        colors = ['red' if x == reference else 'lightblue' for x in array]
        fig.add_trace(
            go.Bar(
                x=list(range(len(array))),
                y=array,
                marker_color=colors,
                text=array,
                textposition='auto',
                name='Array Elements'
            ),
            row=1, col=1
        )
        
        # Hash map visualization as table
        hash_data = []
        for diff, info in sorted(count_map.items()):
            hash_data.append([
                f"Diff: {diff}",
                f"Values: {info['values']}",
                f"Count: {info['count']}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Difference', 'Values', 'Count'],
                           fill_color='lightgray',
                           align='left'),
                cells=dict(values=list(zip(*hash_data)) if hash_data else [[], [], []],
                          fill_color='white',
                          align='left')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Hash Map Creation (Reference: {reference})',
            showlegend=False,
            height=600
        )
        
        fig.update_xaxes(title_text="Index", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        
        return fig
    
    def _visualize_final_optimized_step(self, step: dict) -> go.Figure:
        """Visualize final optimized result step."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sorted Differences Order', 'Final Sorted Array'),
            vertical_spacing=0.3
        )
        
        sorted_diffs = step['sorted_differences']
        sorted_array = step['sorted_array']
        count_map = step['count_map']
        
        # Show sorted differences
        diff_counts = [count_map[diff]['count'] for diff in sorted_diffs]
        fig.add_trace(
            go.Bar(
                x=sorted_diffs,
                y=diff_counts,
                name='Difference Counts',
                marker_color='lightcoral',
                text=diff_counts,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Show final sorted array
        fig.add_trace(
            go.Bar(
                x=list(range(len(sorted_array))),
                y=sorted_array,
                name='Final Sorted Array',
                marker_color='lightgreen',
                text=sorted_array,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Final Optimized Result',
            showlegend=False,
            height=500
        )
        
        fig.update_xaxes(title_text="Difference Value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Index", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        return fig
