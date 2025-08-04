"""
Feature and graph encoders for DGDM model.

Implements various encoding strategies for histopathology features
and graph-structured data processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, List, Optional, Union
import math

from dgdm_histopath.core.graph_layers import DynamicGraphLayer, GraphConvolution
from dgdm_histopath.core.attention import MultiHeadAttention


class FeatureEncoder(nn.Module):
    """
    Feature encoder for node-level histopathology features.
    
    Encodes patch-level features into a common embedding space
    with optional normalization and regularization.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalization: str = "layer",
        use_residual: bool = True
    ):
        """
        Initialize feature encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of encoding layers
            dropout: Dropout probability
            activation: Activation function
            normalization: Normalization type
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # Normalization layers
        norm_layer = self._get_norm_layer(normalization, hidden_dim)
        
        # Build encoder layers
        layers = []
        
        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(norm_layer())
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(norm_layer())
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            
        self.encoder = nn.Sequential(*layers)
        
        # Residual projection if input and output dims differ
        if use_residual and input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None
            
    def _get_norm_layer(self, normalization: str, dim: int):
        """Get normalization layer class."""
        if normalization == "layer":
            return lambda: nn.LayerNorm(dim)
        elif normalization == "batch":
            return lambda: nn.BatchNorm1d(dim)
        elif normalization == "instance":
            return lambda: nn.InstanceNorm1d(dim)
        else:
            return lambda: nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature encoder.
        
        Args:
            x: Input features [num_nodes, input_dim]
            
        Returns:
            Encoded features [num_nodes, hidden_dim]
        """
        encoded = self.encoder(x)
        
        # Apply residual connection if enabled
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            encoded = encoded + residual
            
        return encoded


class GraphEncoder(nn.Module):
    """
    Graph encoder using dynamic graph convolutions and attention.
    
    Processes graph-structured histopathology data with multiple
    graph convolution layers and attention mechanisms.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_layers: int = 4,
        attention_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        normalization: str = "layer",
        use_edge_features: bool = True,
        aggregation: str = "add"
    ):
        """
        Initialize graph encoder.
        
        Args:
            input_dim: Input node feature dimension
            hidden_dims: List of hidden dimensions for each layer
            num_layers: Number of graph convolution layers
            attention_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function
            normalization: Normalization type
            use_edge_features: Whether to use edge features
            aggregation: Graph aggregation method
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.use_edge_features = use_edge_features
        
        # Build graph layers
        self.graph_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # Determine dimensions for each layer
        dims = [input_dim] + hidden_dims
        
        for i in range(num_layers):
            in_dim = dims[i]
            out_dim = dims[min(i + 1, len(dims) - 1)]
            
            # Graph convolution layer
            if use_edge_features:
                graph_layer = DynamicGraphLayer(
                    node_dim=in_dim,
                    edge_dim=32,  # Assumed edge feature dimension
                    hidden_dim=out_dim,
                    num_heads=attention_heads,
                    dropout=dropout
                )
            else:
                graph_layer = GraphConvolution(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    edge_dim=None
                )
                
            self.graph_layers.append(graph_layer)
            
            # Normalization layer
            norm_layer = self._get_norm_layer(normalization, out_dim)
            self.norm_layers.append(norm_layer)
            
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final output projection
        self.output_proj = nn.Linear(dims[-1], dims[-1])
        
    def _get_norm_layer(self, normalization: str, dim: int) -> nn.Module:
        """Get normalization layer."""
        if normalization == "layer":
            return nn.LayerNorm(dim)
        elif normalization == "batch":
            return nn.BatchNorm1d(dim)
        elif normalization == "instance":
            return nn.InstanceNorm1d(dim)
        else:
            return nn.Identity()
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through graph encoder.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_dim]
            batch: Batch indices for nodes
            
        Returns:
            Dictionary with encoded embeddings and intermediate representations
        """
        embeddings = x
        layer_outputs = []
        
        for i, (graph_layer, norm_layer) in enumerate(zip(self.graph_layers, self.norm_layers)):
            # Apply graph layer
            if isinstance(graph_layer, DynamicGraphLayer):
                # Use edge attributes if available
                if edge_attr is not None and self.use_edge_features:
                    embeddings = graph_layer(embeddings, edge_index, edge_attr)
                else:
                    # Create dummy edge attributes
                    dummy_edge_attr = torch.zeros(
                        edge_index.size(1), 32, device=embeddings.device
                    )
                    embeddings = graph_layer(embeddings, edge_index, dummy_edge_attr)
            else:
                # Standard graph convolution
                embeddings = graph_layer(embeddings, edge_index, edge_attr)
                
            # Apply normalization and activation
            embeddings = norm_layer(embeddings)
            embeddings = self.activation(embeddings)
            embeddings = self.dropout(embeddings)
            
            layer_outputs.append(embeddings.clone())
            
        # Final output projection
        final_embeddings = self.output_proj(embeddings)
        
        return {
            "embeddings": final_embeddings,
            "layer_outputs": layer_outputs,
            "num_nodes": x.size(0)
        }


class PositionalEncoder(nn.Module):
    """
    Positional encoder for spatial coordinates in histopathology.
    
    Encodes 2D spatial coordinates using sinusoidal embeddings
    similar to transformer positional encodings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 10000,
        temperature: float = 1.0
    ):
        """
        Initialize positional encoder.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            temperature: Temperature scaling factor
        """
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        
        # Create positional encoding table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Generate positional encodings for 2D coordinates.
        
        Args:
            positions: 2D coordinates [batch_size, 2] or [num_nodes, 2]
            
        Returns:
            Positional encodings [batch_size, d_model] or [num_nodes, d_model]
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
            
        batch_size = positions.size(0)
        pos_enc = torch.zeros(batch_size, self.d_model, device=positions.device)
        
        # Encode x and y coordinates separately
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]
        
        # Normalize coordinates
        x_norm = (x_pos - x_pos.min()) / (x_pos.max() - x_pos.min() + 1e-8)
        y_norm = (y_pos - y_pos.min()) / (y_pos.max() - y_pos.min() + 1e-8)
        
        # Scale to match positional encoding range
        x_scaled = (x_norm * self.pe.size(0)).long().clamp(0, self.pe.size(0) - 1)
        y_scaled = (y_norm * self.pe.size(0)).long().clamp(0, self.pe.size(0) - 1)
        
        # Get positional encodings
        x_enc = self.pe[x_scaled]
        y_enc = self.pe[y_scaled]
        
        # Combine x and y encodings
        pos_enc[:, :self.d_model//2] = x_enc[:, :self.d_model//2]
        pos_enc[:, self.d_model//2:] = y_enc[:, :self.d_model//2]
        
        return pos_enc / self.temperature


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder for multi-scale histopathology analysis.
    
    Processes tissue graphs at multiple resolution levels and
    combines information across scales.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_levels: int = 3,
        level_dims: Optional[List[int]] = None,
        pooling_ratios: Optional[List[float]] = None,
        cross_level_attention: bool = True
    ):
        """
        Initialize hierarchical encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_levels: Number of hierarchical levels
            level_dims: Dimensions for each level
            pooling_ratios: Pooling ratios between levels
            cross_level_attention: Whether to use cross-level attention
        """
        super().__init__()
        
        self.num_levels = num_levels
        self.cross_level_attention = cross_level_attention
        
        if level_dims is None:
            level_dims = [hidden_dim] * num_levels
            
        if pooling_ratios is None:
            pooling_ratios = [0.5] * (num_levels - 1)
            
        # Level-specific encoders
        self.level_encoders = nn.ModuleList()
        for i in range(num_levels):
            encoder = GraphEncoder(
                input_dim=input_dim if i == 0 else level_dims[i-1],
                hidden_dims=[level_dims[i]],
                num_layers=2
            )
            self.level_encoders.append(encoder)
            
        # Cross-level attention if enabled
        if cross_level_attention:
            self.cross_attention = nn.ModuleList()
            for i in range(num_levels - 1):
                attention = MultiHeadAttention(
                    embed_dim=level_dims[i],
                    num_heads=8
                )
                self.cross_attention.append(attention)
                
        # Final fusion layer
        total_dim = sum(level_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        hierarchical_graphs: List[torch.Tensor],
        edge_indices: List[torch.Tensor],
        edge_attrs: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical encoder.
        
        Args:
            hierarchical_graphs: List of graph representations at different levels
            edge_indices: List of edge indices for each level
            edge_attrs: Optional list of edge attributes
            
        Returns:
            Fused hierarchical representation
        """
        level_outputs = []
        
        # Process each level
        for i, (graph, edge_index) in enumerate(zip(hierarchical_graphs, edge_indices)):
            edge_attr = edge_attrs[i] if edge_attrs else None
            
            # Encode at current level
            encoded = self.level_encoders[i](graph, edge_index, edge_attr)
            level_outputs.append(encoded["embeddings"])
            
        # Apply cross-level attention if enabled
        if self.cross_level_attention and len(level_outputs) > 1:
            attended_outputs = [level_outputs[0]]
            
            for i in range(1, len(level_outputs)):
                # Attend from current level to previous level
                curr_level = level_outputs[i].unsqueeze(0)
                prev_level = level_outputs[i-1].unsqueeze(0)
                
                attended, _ = self.cross_attention[i-1](curr_level, prev_level, prev_level)
                attended_outputs.append(attended.squeeze(0))
                
            level_outputs = attended_outputs
            
        # Pool each level to same size (use mean pooling for simplicity)
        pooled_outputs = []
        for output in level_outputs:
            pooled = output.mean(dim=0, keepdim=True)
            pooled_outputs.append(pooled)
            
        # Concatenate and fuse
        concatenated = torch.cat(pooled_outputs, dim=-1)
        fused = self.fusion(concatenated)
        
        return fused.squeeze(0)