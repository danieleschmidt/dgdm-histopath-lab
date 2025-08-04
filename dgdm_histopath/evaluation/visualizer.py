"""
Attention visualization and interpretability tools for DGDM.

Provides methods for visualizing attention patterns, creating heatmaps,
and generating interpretable visualizations of model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging

# Optional imports for advanced visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class AttentionVisualizer:
    """
    Visualization tools for DGDM attention patterns and interpretability.
    
    Creates various types of visualizations including attention heatmaps,
    graph visualizations, and overlay images for histopathology analysis.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        style: str = "whitegrid"
    ):
        """
        Initialize attention visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
            dpi: Resolution for saved figures
            style: Seaborn style for plots
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_style(style)
        
        self.logger = logging.getLogger(__name__)
        
    def create_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        patch_positions: Optional[np.ndarray] = None,
        title: str = "Attention Heatmap",
        save_path: Optional[Union[str, Path]] = None,
        interactive: bool = False
    ) -> Union[plt.Figure, Any]:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights [num_nodes] or [num_nodes, num_nodes]
            patch_positions: Spatial positions of patches [num_nodes, 2]
            title: Plot title
            save_path: Path to save the figure
            interactive: Whether to create interactive plot (requires plotly)
            
        Returns:
            Figure object (matplotlib or plotly)
        """
        
        if attention_weights.ndim == 2:
            # Matrix attention - use diagonal or sum
            attention_scores = np.diag(attention_weights)
        else:
            attention_scores = attention_weights
            
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_heatmap(
                attention_scores, patch_positions, title, save_path
            )
        else:
            return self._create_static_heatmap(
                attention_scores, patch_positions, title, save_path
            )
            
    def _create_static_heatmap(
        self,
        attention_scores: np.ndarray,
        patch_positions: Optional[np.ndarray],
        title: str,
        save_path: Optional[Union[str, Path]]
    ) -> plt.Figure:
        """Create static matplotlib heatmap."""
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if patch_positions is not None:
            # Scatter plot with attention as color
            scatter = ax.scatter(
                patch_positions[:, 0],
                patch_positions[:, 1],
                c=attention_scores,
                cmap='viridis',
                s=50,
                alpha=0.7
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Attention Weight', rotation=270, labelpad=20)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            
        else:
            # Bar plot of attention scores
            indices = np.arange(len(attention_scores))
            bars = ax.bar(indices, attention_scores, alpha=0.7)
            
            # Color bars by attention value
            colors = plt.cm.viridis(attention_scores / attention_scores.max())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
                
            ax.set_xlabel('Node Index')
            ax.set_ylabel('Attention Weight')
            
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved attention heatmap to {save_path}")
            
        return fig
        
    def _create_interactive_heatmap(
        self,
        attention_scores: np.ndarray,
        patch_positions: Optional[np.ndarray],
        title: str,
        save_path: Optional[Union[str, Path]]
    ) -> go.Figure:
        """Create interactive plotly heatmap."""
        
        if patch_positions is not None:
            # Scatter plot
            fig = go.Figure(data=go.Scatter(
                x=patch_positions[:, 0],
                y=patch_positions[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=attention_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Attention Weight")
                ),
                text=[f"Node {i}: {score:.3f}" for i, score in enumerate(attention_scores)],
                hovertemplate="<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>"
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="X Position",
                yaxis_title="Y Position",
                showlegend=False
            )
            
        else:
            # Bar plot
            fig = go.Figure(data=go.Bar(
                x=list(range(len(attention_scores))),
                y=attention_scores,
                marker=dict(
                    color=attention_scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Attention Weight")
                ),
                hovertemplate="<b>Node %{x}</b><br>Attention: %{y:.3f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Node Index",
                yaxis_title="Attention Weight"
            )
            
        if save_path:
            fig.write_html(str(save_path).replace('.png', '.html'))
            self.logger.info(f"Saved interactive heatmap to {save_path}")
            
        return fig
        
    def create_graph_visualization(
        self,
        graph_data: Dict[str, Any],
        attention_weights: Optional[np.ndarray] = None,
        title: str = "Tissue Graph",
        save_path: Optional[Union[str, Path]] = None,
        layout: str = "spring"
    ) -> plt.Figure:
        """
        Create graph structure visualization.
        
        Args:
            graph_data: Dictionary containing graph information
            attention_weights: Optional attention weights for node coloring
            title: Plot title
            save_path: Path to save the figure
            layout: Graph layout algorithm
            
        Returns:
            Figure object
        """
        
        try:
            import networkx as nx
        except ImportError:
            self.logger.error("NetworkX required for graph visualization")
            return None
            
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        num_nodes = graph_data.get('num_nodes', 0)
        G.add_nodes_from(range(num_nodes))
        
        # Add edges
        if 'edge_index' in graph_data:
            edge_index = graph_data['edge_index']
            if isinstance(edge_index, np.ndarray):
                edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
            else:
                edges = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))]
            G.add_edges_from(edges)
            
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
            
        # Create visualization
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5, edge_color='gray')
        
        # Draw nodes
        if attention_weights is not None:
            # Color nodes by attention
            node_colors = attention_weights[:num_nodes] if len(attention_weights) >= num_nodes else attention_weights
            nodes = nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=node_colors,
                node_size=50,
                cmap='viridis',
                alpha=0.8
            )
            
            # Add colorbar
            cbar = plt.colorbar(nodes, ax=ax)
            cbar.set_label('Attention Weight', rotation=270, labelpad=20)
            
        else:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=50, alpha=0.8)
            
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved graph visualization to {save_path}")
            
        return fig
        
    def create_prediction_summary(
        self,
        prediction: Dict[str, Any],
        title: str = "DGDM Prediction Summary",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Create comprehensive prediction summary visualization.
        
        Args:
            prediction: Prediction dictionary from DGDM model
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        
        # Determine subplot layout based on available data
        subplots = []
        
        if 'classification_probs' in prediction:
            subplots.append('classification')
            
        if 'regression_outputs' in prediction:
            subplots.append('regression')
            
        if 'attention_weights' in prediction:
            subplots.append('attention')
            
        if not subplots:
            self.logger.warning("No visualizable prediction data found")
            return None
            
        # Create subplots
        n_subplots = len(subplots)
        fig, axes = plt.subplots(1, n_subplots, figsize=(5 * n_subplots, 5), dpi=self.dpi)
        
        if n_subplots == 1:
            axes = [axes]
            
        for i, subplot_type in enumerate(subplots):
            ax = axes[i]
            
            if subplot_type == 'classification':
                # Classification probabilities
                probs = prediction['classification_probs']
                classes = [f'Class {i}' for i in range(len(probs))]
                
                bars = ax.bar(classes, probs, alpha=0.7)
                
                # Highlight predicted class
                predicted_class = np.argmax(probs)
                bars[predicted_class].set_color('red')
                
                ax.set_title('Classification Probabilities')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                
                # Add confidence text
                confidence = prediction.get('confidence', np.max(probs))
                ax.text(0.02, 0.98, f'Confidence: {confidence:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
            elif subplot_type == 'regression':
                # Regression outputs
                outputs = prediction['regression_outputs']
                targets = [f'Target {i}' for i in range(len(outputs))]
                
                ax.bar(targets, outputs, alpha=0.7, color='green')
                ax.set_title('Regression Outputs')
                ax.set_ylabel('Value')
                
            elif subplot_type == 'attention':
                # Attention weights histogram
                attention = prediction['attention_weights']
                if attention.ndim == 2:
                    attention = np.diag(attention)
                    
                ax.hist(attention, bins=20, alpha=0.7, color='purple')
                ax.set_title('Attention Weight Distribution')
                ax.set_xlabel('Attention Weight')
                ax.set_ylabel('Frequency')
                
                # Add statistics
                mean_attn = np.mean(attention)
                std_attn = np.std(attention)
                ax.axvline(mean_attn, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_attn:.3f}')
                ax.legend()
                
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved prediction summary to {save_path}")
            
        return fig
        
    def create_biomarker_visualization(
        self,
        biomarkers: Dict[str, Any],
        title: str = "Top Biomarkers",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Visualize extracted biomarkers.
        
        Args:
            biomarkers: Biomarker dictionary from predictor
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        
        if not biomarkers.get('biomarkers'):
            self.logger.warning("No biomarkers to visualize")
            return None
            
        biomarker_list = biomarkers['biomarkers']
        
        # Extract data for plotting
        ranks = [b['rank'] for b in biomarker_list]
        scores = [b['importance_score'] for b in biomarker_list]
        node_indices = [b['node_index'] for b in biomarker_list]
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        bars = ax.barh(ranks, scores, alpha=0.7)
        
        # Color bars by importance
        colors = plt.cm.viridis(np.array(scores) / max(scores))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        # Add node indices as labels
        for i, (rank, score, node_idx) in enumerate(zip(ranks, scores, node_indices)):
            ax.text(score + 0.01 * max(scores), rank, f'Node {node_idx}', 
                   verticalalignment='center', fontsize=8)
            
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Rank')
        ax.set_title(f"{title} (Method: {biomarkers.get('method', 'unknown')})")
        ax.invert_yaxis()  # Highest rank at top
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved biomarker visualization to {save_path}")
            
        return fig
        
    def create_uncertainty_plot(
        self,
        uncertainty: Dict[str, float],
        title: str = "Prediction Uncertainty",
        save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """
        Visualize prediction uncertainty measures.
        
        Args:
            uncertainty: Uncertainty dictionary
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Figure object
        """
        
        # Extract uncertainty measures (excluding method)
        measures = {k: v for k, v in uncertainty.items() if k != 'method' and isinstance(v, (int, float))}
        
        if not measures:
            self.logger.warning("No uncertainty measures to visualize")
            return None
            
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        measure_names = list(measures.keys())
        measure_values = list(measures.values())
        
        bars = ax.bar(measure_names, measure_values, alpha=0.7, color='orange')
        
        ax.set_ylabel('Uncertainty Value')
        ax.set_title(f"{title} (Method: {uncertainty.get('method', 'unknown')})")
        
        # Add value labels on bars
        for bar, value in zip(bars, measure_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(measure_values),
                   f'{value:.3f}', ha='center', va='bottom')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved uncertainty plot to {save_path}")
            
        return fig