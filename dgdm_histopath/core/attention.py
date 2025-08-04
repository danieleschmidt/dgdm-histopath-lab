"""
Multi-head attention mechanisms for histopathology analysis.

Implements specialized attention layers for processing tissue graphs,
including spatial attention and cross-modal attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism optimized for histopathology graphs.
    Supports both self-attention and cross-attention with optional masking.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        kdv_bias: bool = True,
        batch_first: bool = True,
        add_zero_attn: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.add_zero_attn = add_zero_attn
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
            
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=kdv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=kdv_bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
            
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim] or [seq_len, batch_size, embed_dim]
            key: Key tensor (defaults to query)
            value: Value tensor (defaults to key)
            key_padding_mask: Mask for padding tokens
            need_weights: Whether to return attention weights
            attn_mask: Attention mask
            average_attn_weights: Whether to average attention weights across heads
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle input shapes
        if not self.batch_first:
            query = query.transpose(0, 1)
            if key is not None:
                key = key.transpose(0, 1)
            if value is not None:
                value = value.transpose(0, 1)
                
        batch_size, seq_len, embed_dim = query.shape
        
        # Use query as key and value if not provided (self-attention)
        if key is None:
            key = query
        if value is None:
            value = key
            
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Add zero attention if specified
        if self.add_zero_attn:
            zero_attn_shape = (batch_size, self.num_heads, 1, self.head_dim)
            k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=2)
            v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=2)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
                
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_scores += attn_mask
                
        # Apply key padding mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.view(batch_size, self.num_heads, seq_len, -1)
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        # Handle output shape
        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)
            
        # Prepare attention weights for output
        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            else:
                attn_weights = attn_weights.view(
                    batch_size * self.num_heads, seq_len, -1
                )
        else:
            attn_weights = None
            
        return attn_output, attn_weights


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for processing tissue patches with spatial awareness.
    Incorporates positional encodings and spatial relationships.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        max_positions: int = 10000,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_positions = max_positions
        self.temperature = temperature
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout=dropout
        )
        
        # Positional encoding for 2D coordinates
        self.pos_encoding = nn.Parameter(
            torch.randn(max_positions, embed_dim) * 0.02
        )
        
        # Spatial relationship encoding
        self.spatial_proj = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def get_positional_encoding(self, positions: Tensor) -> Tensor:
        """
        Get positional encodings for 2D spatial coordinates.
        
        Args:
            positions: Tensor of shape [batch_size, num_patches, 2] containing (x, y) coordinates
            
        Returns:
            Positional encodings of shape [batch_size, num_patches, embed_dim]
        """
        batch_size, num_patches = positions.shape[:2]
        
        # Normalize positions to [0, 1] range
        pos_norm = positions.float()
        if pos_norm.numel() > 0:
            pos_norm = (pos_norm - pos_norm.min()) / (pos_norm.max() - pos_norm.min() + 1e-8)
        
        # Create sinusoidal positional encodings
        pe = torch.zeros(batch_size, num_patches, self.embed_dim, device=positions.device)
        
        # Encode x and y coordinates separately
        div_term = torch.exp(
            torch.arange(0, self.embed_dim // 2, 2, device=positions.device) *
            -(math.log(10000.0) / (self.embed_dim // 2))
        )
        
        # X coordinate encoding
        pe[:, :, 0::4] = torch.sin(pos_norm[:, :, 0:1] * div_term)
        pe[:, :, 1::4] = torch.cos(pos_norm[:, :, 0:1] * div_term)
        
        # Y coordinate encoding
        pe[:, :, 2::4] = torch.sin(pos_norm[:, :, 1:2] * div_term)
        pe[:, :, 3::4] = torch.cos(pos_norm[:, :, 1:2] * div_term)
        
        return pe
        
    def compute_spatial_bias(self, positions: Tensor) -> Tensor:
        """
        Compute spatial attention bias based on patch positions.
        
        Args:
            positions: Spatial coordinates [batch_size, num_patches, 2]
            
        Returns:
            Spatial bias matrix [batch_size, num_patches, num_patches]
        """
        batch_size, num_patches = positions.shape[:2]
        
        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)  # [batch_size, num_patches, 1, 2]
        pos_j = positions.unsqueeze(1)  # [batch_size, 1, num_patches, 2]
        
        # Euclidean distance
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [batch_size, num_patches, num_patches]
        
        # Convert distances to attention bias (closer patches get higher attention)
        spatial_bias = -distances / self.temperature
        
        return spatial_bias
        
    def forward(
        self,
        x: Tensor,
        positions: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for spatial attention.
        
        Args:
            x: Input features [batch_size, num_patches, embed_dim]
            positions: Spatial coordinates [batch_size, num_patches, 2]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output_features, attention_weights)
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Add positional encoding
        pos_enc = self.get_positional_encoding(positions)
        x_with_pos = x + pos_enc
        
        # Compute spatial attention bias
        spatial_bias = self.compute_spatial_bias(positions)
        
        # Combine with existing mask if provided
        if mask is not None:
            attn_mask = mask + spatial_bias
        else:
            attn_mask = spatial_bias
            
        # Apply attention
        attn_output, attn_weights = self.attention(
            x_with_pos,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        # Residual connection and normalization
        output = self.norm(x + attn_output)
        
        return output, attn_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for integrating different types of histological features.
    Can combine morphological, molecular, and clinical data.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        cross_attention: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cross_attention = cross_attention
        
        # Cross-attention layers
        if cross_attention:
            self.cross_attn = MultiHeadAttention(
                embed_dim, num_heads, dropout=dropout
            )
            self.cross_norm = nn.LayerNorm(embed_dim)
            
        # Self-attention layer
        self.self_attn = MultiHeadAttention(
            embed_dim, num_heads, dropout=dropout
        )
        self.self_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        query: Tensor,
        key_value: Optional[Tensor] = None,
        query_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: Query modality features [batch_size, query_len, embed_dim]
            key_value: Key-value modality features [batch_size, kv_len, embed_dim]
            query_mask: Mask for query sequence
            kv_mask: Mask for key-value sequence
            
        Returns:
            Tuple of (output_features, cross_attention_weights)
        """
        cross_attn_weights = None
        
        # Cross-attention (if key_value is provided)
        if self.cross_attention and key_value is not None:
            cross_output, cross_attn_weights = self.cross_attn(
                query=query,
                key=key_value,
                value=key_value,
                key_padding_mask=kv_mask,
                need_weights=True
            )
            query = self.cross_norm(query + cross_output)
            
        # Self-attention
        self_output, _ = self.self_attn(
            query=query,
            key_padding_mask=query_mask,
            need_weights=False
        )
        query = self.self_norm(query + self_output)
        
        # Feed-forward network
        ffn_output = self.ffn(query)
        output = self.ffn_norm(query + ffn_output)
        
        return output, cross_attn_weights


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention with optional temperature scaling.
    Used as building block for more complex attention mechanisms.
    """
    
    def __init__(self, temperature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            q: Query tensor [..., seq_len_q, d_k]
            k: Key tensor [..., seq_len_k, d_k]
            v: Value tensor [..., seq_len_v, d_v]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            attn_scores.masked_fill_(mask == 0, -1e9)
            
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights