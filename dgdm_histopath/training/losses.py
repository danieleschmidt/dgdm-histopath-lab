"""
Loss functions for DGDM training.

Implements specialized loss functions for self-supervised learning
and downstream tasks in histopathology analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DiffusionLoss(nn.Module):
    """
    Loss function for diffusion-based self-supervised learning.
    
    Computes the noise prediction loss for the diffusion process.
    """
    
    def __init__(self, loss_type: str = "mse", reduction: str = "mean"):
        """
        Initialize diffusion loss.
        
        Args:
            loss_type: Type of loss ("mse", "mae", "huber")
            reduction: Reduction method ("mean", "sum", "none")
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
    def forward(
        self,
        predicted_noise: torch.Tensor,
        target_noise: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diffusion loss.
        
        Args:
            predicted_noise: Predicted noise from model
            target_noise: Target noise
            mask: Optional mask for selective loss computation
            
        Returns:
            Diffusion loss
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(predicted_noise, target_noise, reduction="none")
        elif self.loss_type == "mae":
            loss = F.l1_loss(predicted_noise, target_noise, reduction="none")
        elif self.loss_type == "huber":
            loss = F.huber_loss(predicted_noise, target_noise, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised representation learning.
    
    Implements InfoNCE-style contrastive learning for graph nodes.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        similarity_function: str = "cosine",
        reduction: str = "mean"
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature scaling parameter
            similarity_function: Similarity function ("cosine", "dot")
            reduction: Reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.similarity_function = similarity_function
        self.reduction = reduction
        
    def forward(
        self,
        embeddings: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None,
        positive_pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            embeddings: Node embeddings [num_nodes, embed_dim]
            batch_indices: Batch indices for each node
            positive_pairs: Explicit positive pairs [num_pairs, 2]
            
        Returns:
            Contrastive loss
        """
        if positive_pairs is None:
            # Use within-batch positives (same graph)
            return self._batch_contrastive_loss(embeddings, batch_indices)
        else:
            # Use explicit positive pairs
            return self._pair_contrastive_loss(embeddings, positive_pairs)
            
    def _batch_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss using batch-based positives."""
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute similarity matrix
        if self.similarity_function == "cosine":
            similarity_matrix = torch.matmul(embeddings, embeddings.t())
        elif self.similarity_function == "dot":
            similarity_matrix = torch.matmul(embeddings, embeddings.t())
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_function}")
            
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature
        
        # Create positive mask (same batch index)
        batch_indices = batch_indices.unsqueeze(0)
        positive_mask = (batch_indices == batch_indices.t()).float()
        
        # Remove self-similarities
        positive_mask.fill_diagonal_(0)
        
        # Compute loss
        exp_similarities = torch.exp(similarity_matrix)
        
        # Sum of all similarities (denominator)
        sum_exp_similarities = exp_similarities.sum(dim=1, keepdim=True)
        
        # Sum of positive similarities (numerator)
        positive_similarities = exp_similarities * positive_mask
        sum_positive_similarities = positive_similarities.sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        sum_positive_similarities = torch.clamp(sum_positive_similarities, min=1e-8)
        
        # Compute negative log likelihood
        loss = -torch.log(sum_positive_similarities / sum_exp_similarities)
        
        # Only compute loss for nodes that have positives
        valid_mask = positive_mask.sum(dim=1) > 0
        loss = loss[valid_mask]
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
            
    def _pair_contrastive_loss(
        self,
        embeddings: torch.Tensor,
        positive_pairs: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss using explicit positive pairs."""
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        num_pairs = positive_pairs.size(0)
        losses = []
        
        for i in range(num_pairs):
            anchor_idx, positive_idx = positive_pairs[i]
            
            anchor_emb = embeddings[anchor_idx:anchor_idx+1]  # [1, embed_dim]
            
            # Compute similarities with all embeddings
            similarities = torch.matmul(anchor_emb, embeddings.t()) / self.temperature  # [1, num_nodes]
            
            # Softmax to get probabilities
            probs = F.softmax(similarities, dim=1)
            
            # Loss is negative log probability of positive
            loss = -torch.log(probs[0, positive_idx] + 1e-8)
            losses.append(loss)
            
        losses = torch.stack(losses)
        
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class MaskedLanguageModelingLoss(nn.Module):
    """
    Masked language modeling loss for node features.
    
    Predicts masked node features as in BERT-style pretraining.
    """
    
    def __init__(self, vocab_size: int, ignore_index: int = -100):
        """
        Initialize MLM loss.
        
        Args:
            vocab_size: Size of feature vocabulary
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
    def forward(
        self,
        predicted_features: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MLM loss.
        
        Args:
            predicted_features: Predicted features [num_nodes, vocab_size]
            target_features: Target features [num_nodes]
            mask: Mask indicating which nodes were masked [num_nodes]
            
        Returns:
            MLM loss
        """
        # Only compute loss for masked nodes
        masked_predictions = predicted_features[mask]
        masked_targets = target_features[mask]
        
        if masked_predictions.size(0) == 0:
            return torch.tensor(0.0, device=predicted_features.device)
            
        # Cross-entropy loss
        loss = F.cross_entropy(
            masked_predictions,
            masked_targets,
            ignore_index=self.ignore_index
        )
        
        return loss


class GraphReconstructionLoss(nn.Module):
    """
    Graph reconstruction loss for structural self-supervision.
    
    Predicts graph structure from node embeddings.
    """
    
    def __init__(self, loss_type: str = "bce", edge_sampling_ratio: float = 1.0):
        """
        Initialize graph reconstruction loss.
        
        Args:
            loss_type: Type of loss ("bce", "focal")
            edge_sampling_ratio: Ratio of edges to sample for efficiency
        """
        super().__init__()
        self.loss_type = loss_type
        self.edge_sampling_ratio = edge_sampling_ratio
        
    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """
        Compute graph reconstruction loss.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, embed_dim]
            edge_index: True edge indices [2, num_edges]
            num_nodes: Total number of nodes
            
        Returns:
            Graph reconstruction loss
        """
        device = node_embeddings.device
        
        # Sample edges if specified
        num_edges = edge_index.size(1)
        if self.edge_sampling_ratio < 1.0:
            num_sample = int(num_edges * self.edge_sampling_ratio)
            sample_indices = torch.randperm(num_edges)[:num_sample]
            edge_index = edge_index[:, sample_indices]
            
        # Positive edges
        pos_edges = edge_index
        pos_scores = self._compute_edge_scores(node_embeddings, pos_edges)
        pos_labels = torch.ones(pos_scores.size(0), device=device)
        
        # Negative edges (random sampling)
        num_neg = pos_scores.size(0)
        neg_edges = self._sample_negative_edges(num_nodes, num_neg, pos_edges, device)
        neg_scores = self._compute_edge_scores(node_embeddings, neg_edges)
        neg_labels = torch.zeros(neg_scores.size(0), device=device)
        
        # Combine positive and negative
        all_scores = torch.cat([pos_scores, neg_scores])
        all_labels = torch.cat([pos_labels, neg_labels])
        
        # Compute loss
        if self.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
        elif self.loss_type == "focal":
            loss = self._focal_loss(all_scores, all_labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
        return loss
        
    def _compute_edge_scores(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute edge existence scores."""
        
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        
        # Dot product similarity
        scores = torch.sum(source_embeddings * target_embeddings, dim=1)
        
        return scores
        
    def _sample_negative_edges(
        self,
        num_nodes: int,
        num_neg: int,
        pos_edges: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Sample negative edges."""
        
        # Create set of positive edges for efficient lookup
        pos_edge_set = set()
        for i in range(pos_edges.size(1)):
            edge = (pos_edges[0, i].item(), pos_edges[1, i].item())
            pos_edge_set.add(edge)
            pos_edge_set.add((edge[1], edge[0]))  # Add reverse edge
            
        # Sample negative edges
        neg_edges = []
        max_attempts = num_neg * 10  # Avoid infinite loop
        attempts = 0
        
        while len(neg_edges) < num_neg and attempts < max_attempts:
            source = torch.randint(0, num_nodes, (1,)).item()
            target = torch.randint(0, num_nodes, (1,)).item()
            
            if source != target and (source, target) not in pos_edge_set:
                neg_edges.append([source, target])
                
            attempts += 1
            
        # Fill remaining with random edges if needed
        while len(neg_edges) < num_neg:
            source = torch.randint(0, num_nodes, (1,)).item()
            target = torch.randint(0, num_nodes, (1,)).item()
            if source != target:
                neg_edges.append([source, target])
                
        return torch.tensor(neg_edges, device=device).t()
        
    def _focal_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Compute focal loss for imbalanced classification."""
        
        probs = torch.sigmoid(scores)
        ce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="none")
        
        p_t = probs * labels + (1 - probs) * (1 - labels)
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        
        focal_loss = alpha_t * (1 - p_t) ** gamma * ce_loss
        
        return focal_loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with automatic task weighting.
    
    Combines multiple losses with learnable task-specific weights.
    """
    
    def __init__(self, num_tasks: int, use_uncertainty_weighting: bool = True):
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks
            use_uncertainty_weighting: Whether to use uncertainty-based weighting
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable log variance parameters
            self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        else:
            self.log_vars = None
            
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted multi-task loss.
        
        Args:
            losses: Individual task losses [num_tasks]
            
        Returns:
            Weighted total loss
        """
        if self.use_uncertainty_weighting and self.log_vars is not None:
            # Uncertainty-based weighting
            precision = torch.exp(-self.log_vars)
            weighted_losses = precision * losses + self.log_vars
            return weighted_losses.sum()
        else:
            # Simple average
            return losses.mean()