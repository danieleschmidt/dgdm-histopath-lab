"""
Graph neural network layers for processing tissue graphs in histopathology.

Implements dynamic graph convolutions that adapt to tissue morphology and
spatial relationships between tissue patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Optional, Tuple, Union
import math


class GraphConvolution(MessagePassing):
    """
    Basic graph convolution layer with edge feature integration.
    Adapted for histopathology tissue graphs.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        # Node transformation
        self.node_lin = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge transformation if edge features are provided
        if edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, out_channels, bias=False)
        else:
            self.edge_lin = None
            
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.node_lin.weight)
        if self.edge_lin is not None:
            nn.init.xavier_uniform_(self.edge_lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
            
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        """Forward pass."""
        if self.normalize and isinstance(edge_index, Tensor):
            if self.add_self_loops:
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
                
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = None
            
        # Transform node features
        x = self.node_lin(x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm, size=size)
        
        if self.bias is not None:
            out += self.bias
            
        return out
        
    def message(self, x_j: Tensor, edge_attr: OptTensor, norm: OptTensor) -> Tensor:
        """Construct messages."""
        msg = x_j
        
        if edge_attr is not None and self.edge_lin is not None:
            edge_features = self.edge_lin(edge_attr)
            msg = msg + edge_features
            
        if norm is not None:
            msg = norm.view(-1, 1) * msg
            
        return msg


class DynamicGraphLayer(nn.Module):
    """
    Dynamic graph layer that adapts connectivity based on tissue morphology.
    Uses attention mechanisms to dynamically weight edge connections.
    """
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention for dynamic edge weighting
        self.node_to_qkv = nn.Linear(node_dim, hidden_dim * 3)
        self.edge_to_key = nn.Linear(edge_dim, hidden_dim)
        
        # Graph convolution layers
        self.graph_conv1 = GraphConvolution(node_dim, hidden_dim, edge_dim)
        self.graph_conv2 = GraphConvolution(hidden_dim, hidden_dim, edge_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, node_dim)
        
        # Normalization and regularization
        self.dropout = nn.Dropout(dropout)
        if use_layer_norm:
            self.norm1 = nn.LayerNorm(node_dim)
            self.norm2 = nn.LayerNorm(node_dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
        # Activation
        self.activation = nn.GELU()
        
    def compute_dynamic_edges(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Compute dynamic edge weights based on node and edge features.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Dynamic edge weights [num_edges, num_heads]
        """
        row, col = edge_index
        
        # Compute Q, K, V from node features
        qkv = self.node_to_qkv(x)  # [num_nodes, hidden_dim * 3]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        k = k.view(-1, self.num_heads, self.head_dim)
        
        # Get node pairs for edges
        q_edges = q[row]  # [num_edges, num_heads, head_dim]
        k_edges = k[col]  # [num_edges, num_heads, head_dim]
        
        # Include edge features in attention computation
        edge_k = self.edge_to_key(edge_attr)  # [num_edges, hidden_dim]
        edge_k = edge_k.view(-1, self.num_heads, self.head_dim)
        
        # Combine node and edge keys
        combined_k = k_edges + edge_k
        
        # Compute attention scores
        attn_scores = torch.sum(q_edges * combined_k, dim=-1)  # [num_edges, num_heads]
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        
        # Apply softmax per node (normalize over incoming edges)
        attn_weights = softmax(attn_scores, col, num_nodes=x.size(0))
        
        return attn_weights
        
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor
    ) -> Tensor:
        """
        Forward pass with dynamic edge weighting.
        
        Args:
            x: Node features [num_nodes, node_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, edge_dim]
            
        Returns:
            Updated node features [num_nodes, node_dim]
        """
        residual = x
        
        # Compute dynamic edge weights
        edge_weights = self.compute_dynamic_edges(x, edge_index, edge_attr)
        
        # Average edge weights across heads for convolution
        edge_weights_avg = edge_weights.mean(dim=1)  # [num_edges]
        
        # Apply graph convolutions with dynamic weighting
        h1 = self.graph_conv1(x, edge_index, edge_attr)
        h1 = self.activation(h1)
        h1 = self.dropout(h1)
        
        h2 = self.graph_conv2(h1, edge_index, edge_attr)
        h2 = self.activation(h2)
        h2 = self.dropout(h2)
        
        # Output projection
        out = self.output_proj(h2)
        
        # Residual connection and normalization
        out = self.norm1(out + residual)
        
        return out


class AdaptiveGraphPooling(nn.Module):
    """
    Adaptive graph pooling that can coarsen graphs based on tissue hierarchies.
    Uses learned cluster assignments for hierarchical pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: str = 'tanh'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        
        # Score computation for node importance
        self.score_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1)
        )
        
        # Nonlinearity for scores
        if nonlinearity == 'tanh':
            self.nonlinearity = torch.tanh
        elif nonlinearity == 'softmax':
            self.nonlinearity = lambda x: torch.softmax(x, dim=0)
        else:
            self.nonlinearity = torch.sigmoid
            
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        """
        Forward pass for adaptive pooling.
        
        Returns:
            Tuple of (pooled_x, pooled_edge_index, pooled_edge_attr, perm)
        """
        # Compute node importance scores
        scores = self.score_net(x).squeeze(-1)  # [num_nodes]
        scores = self.nonlinearity(scores)
        
        # Determine pooling based on ratio or minimum score
        if self.min_score is not None:
            mask = scores >= self.min_score
        else:
            num_nodes = x.size(0)
            k = int(self.ratio * num_nodes)
            k = max(1, k)  # Ensure at least one node remains
            
            _, perm = torch.topk(scores, k, sorted=False)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[perm] = True
            
        # Apply pooling
        perm = mask.nonzero(as_tuple=False).squeeze(-1)
        pooled_x = x[perm] * scores[perm].unsqueeze(-1) * self.multiplier
        
        # Update edge indices
        node_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
        node_map[perm] = torch.arange(perm.size(0), device=x.device)
        
        edge_mask = (node_map[edge_index[0]] >= 0) & (node_map[edge_index[1]] >= 0)
        pooled_edge_index = edge_index[:, edge_mask]
        pooled_edge_index = node_map[pooled_edge_index]
        
        # Update edge attributes if provided
        pooled_edge_attr = edge_attr[edge_mask] if edge_attr is not None else None
        
        return pooled_x, pooled_edge_index, pooled_edge_attr, perm


class GraphUNet(nn.Module):
    """
    U-Net style architecture for graphs with skip connections.
    Useful for multi-scale tissue analysis in histopathology.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        depth: int = 3,
        pool_ratios: Optional[list] = None,
        sum_res: bool = True,
        act: str = 'relu'
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.sum_res = sum_res
        
        if pool_ratios is None:
            pool_ratios = [0.5] * depth
            
        # Activation function
        if act == 'relu':
            self.act = F.relu
        elif act == 'gelu':
            self.act = F.gelu
        else:
            self.act = F.elu
            
        # Encoder (downsampling path)
        self.down_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # First layer
        self.down_convs.append(
            DynamicGraphLayer(in_channels, hidden_channels, hidden_channels)
        )
        
        for i in range(depth):
            self.down_convs.append(
                DynamicGraphLayer(hidden_channels, hidden_channels, hidden_channels)
            )
            self.pools.append(
                AdaptiveGraphPooling(hidden_channels, ratio=pool_ratios[i])
            )
            
        # Bottom layer
        self.bottom_conv = DynamicGraphLayer(
            hidden_channels, hidden_channels, hidden_channels
        )
        
        # Decoder (upsampling path)
        self.up_convs = nn.ModuleList()
        for i in range(depth):
            self.up_convs.append(
                DynamicGraphLayer(
                    hidden_channels * 2, hidden_channels, hidden_channels
                )
            )
            
        # Final output layer
        self.final_conv = nn.Linear(hidden_channels, out_channels)
        
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through Graph U-Net."""
        # Store skip connections
        xs = []
        edge_indices = [edge_index]
        edge_attrs = [edge_attr]
        perms = []
        
        # Encoder path
        x = self.down_convs[0](x, edge_index, edge_attr)
        xs.append(x)
        
        for i in range(self.depth):
            x = self.act(x)
            x = self.down_convs[i + 1](x, edge_indices[-1], edge_attrs[-1])
            xs.append(x)
            
            # Pooling
            x, edge_index, edge_attr, perm = self.pools[i](
                x, edge_indices[-1], edge_attrs[-1], batch
            )
            edge_indices.append(edge_index)
            edge_attrs.append(edge_attr)
            perms.append(perm)
            
        # Bottom
        x = self.act(x)
        x = self.bottom_conv(x, edge_indices[-1], edge_attrs[-1])
        
        # Decoder path
        for i in range(self.depth):
            j = self.depth - 1 - i
            
            # Upsample (simple node replication)
            perm = perms[j]
            up_x = torch.zeros(
                xs[j + 1].size(0), x.size(1), device=x.device, dtype=x.dtype
            )
            up_x[perm] = x
            
            # Skip connection
            if self.sum_res:
                x = up_x + xs[j + 1]
            else:
                x = torch.cat([up_x, xs[j + 1]], dim=1)
                
            x = self.act(x)
            x = self.up_convs[i](x, edge_indices[j + 1], edge_attrs[j + 1])
            
        # Final output
        x = self.final_conv(x)
        
        return x