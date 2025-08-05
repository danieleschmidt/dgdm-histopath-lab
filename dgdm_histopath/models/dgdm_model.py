"""
Dynamic Graph Diffusion Model (DGDM) for histopathology analysis.

Main model class that combines graph neural networks with diffusion processes
for self-supervised learning on tissue graphs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import warnings
from contextlib import contextmanager

from dgdm_histopath.core.diffusion import DiffusionLayer
from dgdm_histopath.core.graph_layers import DynamicGraphLayer, GraphUNet
from dgdm_histopath.core.attention import MultiHeadAttention, SpatialAttention
from dgdm_histopath.models.encoders import FeatureEncoder, GraphEncoder
from dgdm_histopath.models.decoders import ClassificationHead, RegressionHead
from dgdm_histopath.utils.validation import InputValidator, ValidationError
from dgdm_histopath.utils.monitoring import monitor_operation
from dgdm_histopath.utils.logging import get_logger


class ModelConfigurationError(Exception):
    """Exception raised for model configuration errors."""
    pass


class ModelInferenceError(Exception):
    """Exception raised during model inference."""
    pass


class DGDMModel(nn.Module):
    """
    Dynamic Graph Diffusion Model for whole-slide histopathology analysis.
    
    Combines self-supervised diffusion processes with graph neural networks
    to learn robust representations of tissue morphology and spatial relationships.
    """
    
    def __init__(
        self,
        node_features: int = 768,
        hidden_dims: List[int] = [512, 256, 128],
        num_diffusion_steps: int = 10,
        attention_heads: int = 8,
        dropout: float = 0.1,
        graph_layers: int = 4,
        use_spatial_attention: bool = True,
        use_hierarchical: bool = True,
        diffusion_schedule: str = "cosine",
        activation: str = "gelu",
        normalization: str = "layer",
        pooling: str = "attention",
        num_classes: Optional[int] = None,
        regression_targets: int = 0
    ):
        """
        Initialize DGDM model.
        
        Args:
            node_features: Dimension of input node features
            hidden_dims: List of hidden dimensions for graph layers
            num_diffusion_steps: Number of diffusion timesteps
            attention_heads: Number of attention heads
            dropout: Dropout probability
            graph_layers: Number of graph convolution layers
            use_spatial_attention: Whether to use spatial attention
            use_hierarchical: Whether to use hierarchical processing
            diffusion_schedule: Diffusion noise schedule ("linear", "cosine", "sigmoid")
            activation: Activation function ("relu", "gelu", "elu")
            normalization: Normalization type ("layer", "batch", "instance")
            pooling: Graph pooling method ("mean", "max", "attention", "set2set")
            num_classes: Number of classification classes (None for no classification)
            regression_targets: Number of regression targets
        """
        super().__init__()
        
        self.logger = get_logger(__name__)
        
        # Comprehensive input validation
        try:
            self._validate_configuration(
                node_features, hidden_dims, num_diffusion_steps, attention_heads,
                dropout, graph_layers, diffusion_schedule, activation, 
                normalization, pooling, num_classes, regression_targets
            )
        except Exception as e:
            self.logger.error(f"Model configuration validation failed: {e}")
            raise ModelConfigurationError(f"Invalid model configuration: {e}")
        
        # Store validated parameters
        self.node_features = node_features
        self.hidden_dims = hidden_dims
        self.num_diffusion_steps = num_diffusion_steps
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.use_spatial_attention = use_spatial_attention
        self.use_hierarchical = use_hierarchical
        self.pooling = pooling
        self.num_classes = num_classes
        self.regression_targets = regression_targets
        
        self.logger.info(f"Initializing DGDM model with {sum(p.numel() for p in self.parameters() if p.requires_grad)} parameters")
        
        # Input feature encoder
        self.feature_encoder = FeatureEncoder(
            input_dim=node_features,
            hidden_dim=hidden_dims[0],
            dropout=dropout,
            activation=activation,
            normalization=normalization
        )
        
        # Graph encoder with multiple layers
        self.graph_encoder = GraphEncoder(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims,
            num_layers=graph_layers,
            attention_heads=attention_heads,
            dropout=dropout,
            activation=activation,
            normalization=normalization
        )
        
        # Diffusion layer for self-supervised learning
        self.diffusion_layer = DiffusionLayer(
            node_dim=hidden_dims[-1],
            hidden_dim=hidden_dims[-1] * 2,
            num_timesteps=num_diffusion_steps,
            schedule=diffusion_schedule
        )
        
        # Spatial attention if enabled
        if use_spatial_attention:
            self.spatial_attention = SpatialAttention(
                embed_dim=hidden_dims[-1],
                num_heads=attention_heads,
                dropout=dropout
            )
        else:
            self.spatial_attention = None
            
        # Hierarchical processing if enabled
        if use_hierarchical:
            self.hierarchical_processor = GraphUNet(
                in_channels=hidden_dims[-1],
                hidden_channels=hidden_dims[-1],
                out_channels=hidden_dims[-1],
                depth=3
            )
        else:
            self.hierarchical_processor = None
            
        # Graph-level pooling
        self.global_pool = self._create_pooling_layer(
            pooling, hidden_dims[-1], attention_heads
        )
        
        # Task-specific heads
        self.classification_head = None
        self.regression_head = None
        
        if num_classes is not None:
            self.classification_head = ClassificationHead(
                input_dim=hidden_dims[-1],
                num_classes=num_classes,
                hidden_dims=[hidden_dims[-1] // 2],
                dropout=dropout,
                activation=activation
            )
            
        if regression_targets > 0:
            self.regression_head = RegressionHead(
                input_dim=hidden_dims[-1],
                num_targets=regression_targets,
                hidden_dims=[hidden_dims[-1] // 2],
                dropout=dropout,
                activation=activation
            )
            
        # Initialize parameters with monitoring
        with monitor_operation("model_parameter_initialization"):
            self.apply(self._init_weights)
            
        self.logger.info("DGDM model initialization completed successfully")
    
    def _validate_configuration(
        self, node_features: int, hidden_dims: List[int], num_diffusion_steps: int,
        attention_heads: int, dropout: float, graph_layers: int, 
        diffusion_schedule: str, activation: str, normalization: str,
        pooling: str, num_classes: Optional[int], regression_targets: int
    ):
        """Validate model configuration parameters."""
        
        # Validate node features
        InputValidator.validate_integer(node_features, min_val=1, max_val=10000)
        
        # Validate hidden dimensions
        if not isinstance(hidden_dims, list) or len(hidden_dims) == 0:
            raise ValidationError("hidden_dims must be a non-empty list")
        
        for i, dim in enumerate(hidden_dims):
            InputValidator.validate_integer(dim, min_val=1, max_val=10000)
            
        # Ensure dimensions are decreasing or stable
        for i in range(1, len(hidden_dims)):
            if hidden_dims[i] > hidden_dims[i-1]:
                warnings.warn(f"Hidden dimension {i} ({hidden_dims[i]}) > previous ({hidden_dims[i-1]})")
        
        # Validate diffusion parameters
        InputValidator.validate_integer(num_diffusion_steps, min_val=1, max_val=1000)
        InputValidator.validate_enum(diffusion_schedule, ["linear", "cosine", "sigmoid"])
        
        # Validate attention parameters  
        InputValidator.validate_integer(attention_heads, min_val=1, max_val=32)
        if hidden_dims[-1] % attention_heads != 0:
            raise ValidationError(f"Hidden dim {hidden_dims[-1]} not divisible by attention heads {attention_heads}")
        
        # Validate dropout
        InputValidator.validate_numeric(dropout, min_val=0.0, max_val=0.9)
        
        # Validate graph layers
        InputValidator.validate_integer(graph_layers, min_val=1, max_val=20)
        
        # Validate choices
        InputValidator.validate_enum(activation, ["relu", "gelu", "elu", "swish"])
        InputValidator.validate_enum(normalization, ["layer", "batch", "instance", "graph"])
        InputValidator.validate_enum(pooling, ["mean", "max", "attention", "set2set", "sort"])
        
        # Validate task parameters
        if num_classes is not None:
            InputValidator.validate_integer(num_classes, min_val=2, max_val=1000)
            
        InputValidator.validate_integer(regression_targets, min_val=0, max_val=100)
        
        if num_classes is None and regression_targets == 0:
            warnings.warn("No classification or regression targets specified - model will only do pretraining")
    
    def _create_pooling_layer(
        self, pooling: str, hidden_dim: int, attention_heads: int
    ) -> nn.Module:
        """Create global pooling layer."""
        if pooling == "mean":
            return GlobalMeanPool()
        elif pooling == "max":
            return GlobalMaxPool()
        elif pooling == "attention":
            return GlobalAttentionPool(hidden_dim, attention_heads)
        elif pooling == "set2set":
            return GlobalSet2SetPool(hidden_dim)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
            
    def forward(
        self,
        data: Union[Data, Batch],
        mode: str = "inference",
        return_attention: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DGDM model with comprehensive error handling.
        
        Args:
            data: Graph data (single graph or batch)
            mode: Forward mode ("inference", "pretrain", "finetune")
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return node embeddings
            
        Returns:
            Dictionary containing model outputs
            
        Raises:
            ModelInferenceError: If forward pass fails
            ValidationError: If input validation fails
        """
        
        # Input validation and monitoring
        with monitor_operation(f"dgdm_forward_{mode}"):
            try:
                self._validate_forward_inputs(data, mode, return_attention, return_embeddings)
            except Exception as e:
                self.logger.error(f"Forward input validation failed: {e}")
                raise ModelInferenceError(f"Input validation failed: {e}")
                
            outputs = {}
            
            try:
                # Feature encoding with error handling
                with monitor_operation("feature_encoding"):
                    self._validate_input_features(data.x)
                    node_features = self.feature_encoder(data.x)
                    
            except Exception as e:
                self.logger.error(f"Feature encoding failed: {e}")
                raise ModelInferenceError(f"Feature encoding failed: {e}")
            
            try:
                # Graph encoding with error handling  
                with monitor_operation("graph_encoding"):
                    graph_outputs = self.graph_encoder(
                        node_features, data.edge_index, data.edge_attr, data.batch
                    )
                    node_embeddings = graph_outputs["embeddings"]
                    
            except Exception as e:
                self.logger.error(f"Graph encoding failed: {e}")
                raise ModelInferenceError(f"Graph encoding failed: {e}")
            
            try:
                # Continue with the rest of the forward pass
                return self._forward_continue(node_embeddings, data, mode, return_attention, return_embeddings, outputs)
                
            except Exception as e:
                self.logger.error(f"Forward pass failed: {e}")
                raise ModelInferenceError(f"Forward pass failed: {e}")
    
    def _forward_continue(self, node_embeddings, data, mode, return_attention, return_embeddings, outputs):
        
        # Spatial attention if enabled
        attention_weights = None
        if self.spatial_attention is not None and hasattr(data, 'pos'):
            # Group by batch for spatial attention
            if hasattr(data, 'batch'):
                batch_size = data.batch.max().item() + 1
                spatial_outputs = []
                spatial_attention_weights = []
                
                for i in range(batch_size):
                    batch_mask = data.batch == i
                    batch_embeddings = node_embeddings[batch_mask]
                    batch_positions = data.pos[batch_mask]
                    
                    spatial_out, spatial_attn = self.spatial_attention(
                        batch_embeddings.unsqueeze(0),
                        batch_positions.unsqueeze(0)
                    )
                    
                    spatial_outputs.append(spatial_out.squeeze(0))
                    spatial_attention_weights.append(spatial_attn.squeeze(0))
                    
                node_embeddings = torch.cat(spatial_outputs, dim=0)
                if return_attention:
                    attention_weights = spatial_attention_weights
            else:
                node_embeddings, attention_weights = self.spatial_attention(
                    node_embeddings.unsqueeze(0), data.pos.unsqueeze(0)
                )
                node_embeddings = node_embeddings.squeeze(0)
                if return_attention:
                    attention_weights = attention_weights.squeeze(0)
                    
        # Hierarchical processing if enabled
        if self.hierarchical_processor is not None:
            node_embeddings = self.hierarchical_processor(
                node_embeddings, data.edge_index, data.edge_attr, data.batch
            )
            
        # Diffusion processing for self-supervised learning
        if mode == "pretrain":
            diffusion_outputs = self._compute_diffusion_loss(node_embeddings, data)
            outputs.update(diffusion_outputs)
            
        # Global pooling
        graph_embedding = self.global_pool(node_embeddings, data.batch)
        
        # Task-specific predictions
        if self.classification_head is not None and mode in ["inference", "finetune"]:
            classification_logits = self.classification_head(graph_embedding)
            outputs["classification_logits"] = classification_logits
            outputs["classification_probs"] = F.softmax(classification_logits, dim=-1)
            
        if self.regression_head is not None and mode in ["inference", "finetune"]:
            regression_outputs = self.regression_head(graph_embedding)
            outputs["regression_outputs"] = regression_outputs
            
        # Additional outputs
        outputs["graph_embedding"] = graph_embedding
        
        if return_embeddings:
            outputs["node_embeddings"] = node_embeddings
            
        if return_attention and attention_weights is not None:
            outputs["attention_weights"] = attention_weights
            
        return outputs
        
    def _compute_diffusion_loss(
        self, node_embeddings: torch.Tensor, data: Union[Data, Batch]
    ) -> Dict[str, torch.Tensor]:
        """Compute diffusion-based self-supervised loss."""
        # Sample random timesteps
        batch_size = data.batch.max().item() + 1 if hasattr(data, 'batch') else 1
        timesteps = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), device=node_embeddings.device
        )
        
        # Apply diffusion process
        if hasattr(data, 'batch'):
            # Handle batched data
            diffusion_losses = []
            for i in range(batch_size):
                batch_mask = data.batch == i
                batch_embeddings = node_embeddings[batch_mask]
                batch_timestep = timesteps[i:i+1].repeat(batch_embeddings.size(0))
                
                noisy_embeddings, predicted_noise = self.diffusion_layer(
                    batch_embeddings.unsqueeze(0), batch_timestep[:1]
                )
                
                # Compute loss (noise prediction)
                noise_target = torch.randn_like(batch_embeddings)
                loss = F.mse_loss(predicted_noise.squeeze(0), noise_target)
                diffusion_losses.append(loss)
                
            diffusion_loss = torch.stack(diffusion_losses).mean()
        else:
            # Single graph
            noisy_embeddings, predicted_noise = self.diffusion_layer(
                node_embeddings.unsqueeze(0), timesteps
            )
            noise_target = torch.randn_like(node_embeddings)
            diffusion_loss = F.mse_loss(predicted_noise.squeeze(0), noise_target)
            
        return {
            "diffusion_loss": diffusion_loss,
            "noisy_embeddings": noisy_embeddings
        }
        
    def pretrain_step(
        self, data: Union[Data, Batch], mask_ratio: float = 0.15
    ) -> Dict[str, torch.Tensor]:
        """
        Pretraining step with entity masking and diffusion loss.
        
        Args:
            data: Graph data
            mask_ratio: Ratio of nodes to mask
            
        Returns:
            Dictionary with pretraining losses
        """
        # Apply entity masking
        masked_data = self._apply_entity_masking(data, mask_ratio)
        
        # Forward pass in pretrain mode
        outputs = self.forward(masked_data, mode="pretrain")
        
        # Compute reconstruction loss for masked nodes
        if hasattr(data, 'node_mask'):
            reconstruction_loss = self._compute_reconstruction_loss(
                outputs["node_embeddings"], data, masked_data
            )
            outputs["reconstruction_loss"] = reconstruction_loss
            
        # Total pretraining loss
        total_loss = outputs["diffusion_loss"]
        if "reconstruction_loss" in outputs:
            total_loss = total_loss + outputs["reconstruction_loss"]
            
        outputs["total_pretrain_loss"] = total_loss
        
        return outputs
        
    def _apply_entity_masking(
        self, data: Union[Data, Batch], mask_ratio: float
    ) -> Union[Data, Batch]:
        """Apply entity masking for self-supervised pretraining."""
        masked_data = data.clone()
        
        # Create mask for nodes to be masked
        num_nodes = data.x.size(0)
        num_masked = int(num_nodes * mask_ratio)
        
        if num_masked > 0:
            # Random masking
            mask_indices = torch.randperm(num_nodes)[:num_masked]
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
            node_mask[mask_indices] = True
            
            # Replace masked node features with learnable mask token
            mask_token = nn.Parameter(torch.randn(data.x.size(1)))
            masked_data.x = data.x.clone()
            masked_data.x[node_mask] = mask_token
            
            # Store mask for loss computation
            masked_data.node_mask = node_mask
            
        return masked_data
        
    def _compute_reconstruction_loss(
        self, 
        embeddings: torch.Tensor, 
        original_data: Union[Data, Batch], 
        masked_data: Union[Data, Batch]
    ) -> torch.Tensor:
        """Compute reconstruction loss for masked nodes."""
        if not hasattr(masked_data, 'node_mask'):
            return torch.tensor(0.0, device=embeddings.device)
            
        # Get embeddings for masked nodes
        masked_embeddings = embeddings[masked_data.node_mask]
        original_features = original_data.x[masked_data.node_mask]
        
        # Simple reconstruction loss (can be enhanced with more sophisticated decoders)
        reconstruction_loss = F.mse_loss(masked_embeddings, original_features)
        
        return reconstruction_loss
        
    def generate_embeddings(
        self, data: Union[Data, Batch], layer: str = "final"
    ) -> torch.Tensor:
        """
        Generate embeddings at different layers.
        
        Args:
            data: Input graph data
            layer: Which layer to extract embeddings from
            
        Returns:
            Node or graph embeddings
        """
        with torch.no_grad():
            outputs = self.forward(data, mode="inference", return_embeddings=True)
            
            if layer == "final":
                return outputs["graph_embedding"]
            elif layer == "node":
                return outputs["node_embeddings"]
            else:
                raise ValueError(f"Unknown layer: {layer}")


# Global pooling layers
class GlobalMeanPool(nn.Module):
    """Global mean pooling for graphs."""
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                out[i] = x[mask].mean(dim=0)
                
        return out


class GlobalMaxPool(nn.Module):
    """Global max pooling for graphs."""
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            return x.max(dim=0, keepdim=True)[0]
        
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                out[i] = x[mask].max(dim=0)[0]
                
        return out


class GlobalAttentionPool(nn.Module):
    """Global attention-based pooling for graphs."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            # Single graph
            global_token = self.global_token.expand(1, -1, -1)
            x_input = x.unsqueeze(0)
            pooled, _ = self.attention(global_token, x_input, x_input)
            return pooled.squeeze(1)
        
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                batch_x = x[mask].unsqueeze(0)
                global_token = self.global_token.expand(1, -1, -1)
                pooled, _ = self.attention(global_token, batch_x, batch_x)
                out[i] = pooled.squeeze(1)
                
        return out


class GlobalSet2SetPool(nn.Module):
    """Set2Set pooling for graphs."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            # Single graph - simplified Set2Set
            return x.mean(dim=0, keepdim=True)
        
        batch_size = batch.max().item() + 1
        out = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                batch_x = x[mask]
                # Simplified: just use mean (full Set2Set is more complex)
                out[i] = batch_x.mean(dim=0)
                
        return out


# Add validation methods for the DGDM model class - need to be added as methods
def _validate_forward_inputs(self, data, mode, return_attention, return_embeddings):
    """Validate inputs to forward pass."""
    
    # Validate data object
    if not isinstance(data, (Data, Batch)):
        raise ValidationError(f"Expected Data or Batch, got {type(data)}")
        
    # Check required attributes
    required_attrs = ['x', 'edge_index']
    for attr in required_attrs:
        if not hasattr(data, attr):
            raise ValidationError(f"Graph data missing required attribute: {attr}")
            
    # Validate mode
    InputValidator.validate_enum(mode, ["inference", "pretrain", "finetune"])
    
    # Validate boolean flags
    InputValidator.validate_boolean(return_attention)
    InputValidator.validate_boolean(return_embeddings)
    
    # Validate tensor shapes
    if data.x.dim() != 2:
        raise ValidationError(f"Node features must be 2D, got shape {data.x.shape}")
        
    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValidationError(f"Edge index must be 2xN, got shape {data.edge_index.shape}")
        
    # Check for NaN or infinity values
    if torch.isnan(data.x).any():
        raise ValidationError("Node features contain NaN values")
        
    if torch.isinf(data.x).any():
        raise ValidationError("Node features contain infinity values")
        
    # Validate feature dimensions
    if data.x.size(1) != self.node_features:
        raise ValidationError(f"Expected {self.node_features} node features, got {data.x.size(1)}")
        
    # Validate edge indices are within node range
    max_node_idx = data.x.size(0) - 1
    if data.edge_index.max() > max_node_idx:
        raise ValidationError(f"Edge index contains invalid node indices")
        
    if data.edge_index.min() < 0:
        raise ValidationError(f"Edge index contains negative node indices")

def _validate_input_features(self, features):
    """Validate input node features."""
    
    # Check for empty features
    if features.numel() == 0:
        raise ValidationError("Input features are empty")
        
    # Check for extreme values
    if features.abs().max() > 1000:
        warnings.warn(f"Large feature values detected: max={features.abs().max()}")
        
    # Check feature distribution
    if features.std() < 1e-6:
        warnings.warn("Input features have very low variance")
        
    # Memory check
    feature_memory = features.numel() * features.element_size()
    if feature_memory > 1e9:  # 1GB
        warnings.warn(f"Large feature tensor detected: {feature_memory/1e9:.1f}GB")


# These methods need to be added to the DGDMModel class
DGDMModel._validate_forward_inputs = _validate_forward_inputs
DGDMModel._validate_input_features = _validate_input_features