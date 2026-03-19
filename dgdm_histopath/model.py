"""
Graph Diffusion Model for histopathology patch classification.

GraphDiffusionLayer
-------------------
Message-passing layer that diffuses information across the spatial graph.
Uses attention-weighted aggregation:
  1. Compute attention score for each edge  (dot-product of src & dst, scaled).
  2. Softmax over each node's neighbourhood.
  3. Weighted sum of neighbour messages.
  4. Residual connection + layer-norm.

HistoGNN
--------
3-layer GNN built from GraphDiffusionLayer, classifies tissue graphs
as NORMAL (0) or TUMOR (1).

Forward:
  x, edge_index  →  (N, in_features) node features, (2, E) edges
  returns logits of shape (1, 2)  — graph-level classification via mean-pool
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Stable softmax of `scores` grouped by destination node (`dst`).

    Parameters
    ----------
    scores   : (E,)   raw attention logits
    dst      : (E,)   destination node index for each edge
    num_nodes: int

    Returns
    -------
    weights  : (E,)   softmax-normalised attention weights
    """
    # Subtract per-node max for numerical stability
    max_per_node = torch.full((num_nodes,), float("-inf"), device=scores.device)
    max_per_node.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
    shifted = scores - max_per_node[dst]

    exp_scores = shifted.exp()
    denom = torch.zeros(num_nodes, device=scores.device)
    denom.scatter_add_(0, dst, exp_scores)
    denom = denom.clamp(min=1e-9)

    return exp_scores / denom[dst]


def _scatter_sum(values: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Aggregate (E, D) values into (num_nodes, D) by summing per destination."""
    D = values.size(1)
    out = torch.zeros(num_nodes, D, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, dst.unsqueeze(1).expand_as(values), values)
    return out


# ---------------------------------------------------------------------------
# GraphDiffusionLayer
# ---------------------------------------------------------------------------

class GraphDiffusionLayer(nn.Module):
    """
    Single graph diffusion / message-passing layer.

    Attention-weighted neighbourhood aggregation with residual + LayerNorm.

    Parameters
    ----------
    in_dim  : input node feature dimension
    out_dim : output node feature dimension
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Message projections
        self.W_msg = nn.Linear(in_dim, out_dim, bias=False)

        # Attention: score = LeakyReLU( a^T [h_src || h_dst] )
        self.attn_src = nn.Linear(out_dim, 1, bias=False)
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)

        # Residual projection (only if dims differ)
        self.residual = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, in_dim)
        edge_index : (2, E)

        Returns
        -------
        out : (N, out_dim)
        """
        N = x.size(0)
        src_idx, dst_idx = edge_index[0], edge_index[1]   # (E,) each

        # 1. Project all nodes
        h = self.W_msg(x)                                  # (N, out_dim)

        # 2. Attention scores per edge
        a_src = self.attn_src(h[src_idx]).squeeze(-1)      # (E,)
        a_dst = self.attn_dst(h[dst_idx]).squeeze(-1)      # (E,)
        scores = F.leaky_relu(a_src + a_dst, negative_slope=0.2)

        # 3. Normalise per destination
        alpha = _scatter_softmax(scores, dst_idx, N)       # (E,)
        alpha = self.dropout(alpha)

        # 4. Weighted sum of source messages
        messages = h[src_idx] * alpha.unsqueeze(1)         # (E, out_dim)
        agg = _scatter_sum(messages, dst_idx, N)           # (N, out_dim)

        # 5. Residual + norm + activation
        out = self.norm(agg + self.residual(x))
        return self.act(out)


# ---------------------------------------------------------------------------
# HistoGNN
# ---------------------------------------------------------------------------

class HistoGNN(nn.Module):
    """
    3-layer GNN for tissue-graph classification (NORMAL vs TUMOR).

    Architecture:
      GraphDiffusionLayer(in_dim → 64)
      GraphDiffusionLayer(64 → 128)
      GraphDiffusionLayer(128 → 64)
      Mean-pool  →  MLP(64 → 32 → 2)
    """

    def __init__(self, in_dim: int = 4, hidden: int = 64, num_classes: int = 2):
        super().__init__()

        self.conv1 = GraphDiffusionLayer(in_dim, hidden)
        self.conv2 = GraphDiffusionLayer(hidden, hidden * 2)
        self.conv3 = GraphDiffusionLayer(hidden * 2, hidden)

        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x          : (N, in_dim)  node features for ONE graph
        edge_index : (2, E)

        Returns
        -------
        logits : (1, num_classes)
        probs  : (1, num_classes)  softmax probabilities
        """
        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h = self.conv3(h, edge_index)

        # Graph-level mean pooling
        graph_emb = h.mean(dim=0, keepdim=True)   # (1, hidden)

        logits = self.classifier(graph_emb)        # (1, num_classes)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
