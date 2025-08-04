"""
Basic tests for DGDM Histopath Lab components.

Tests core functionality without requiring actual slide data.
"""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data

from dgdm_histopath.models.dgdm_model import DGDMModel
from dgdm_histopath.core.diffusion import DiffusionLayer, DiffusionScheduler
from dgdm_histopath.core.graph_layers import GraphConvolution, DynamicGraphLayer
from dgdm_histopath.core.attention import MultiHeadAttention
from dgdm_histopath.training.losses import DiffusionLoss, ContrastiveLoss


class TestDiffusionComponents:
    """Test diffusion-related components."""
    
    def test_diffusion_scheduler(self):
        """Test diffusion noise scheduler."""
        scheduler = DiffusionScheduler(num_timesteps=100, schedule="cosine")
        
        assert scheduler.betas.shape[0] == 100
        assert torch.all(scheduler.betas > 0)
        assert torch.all(scheduler.betas < 1)
        assert torch.all(scheduler.alphas_cumprod <= 1)
        
    def test_diffusion_layer(self):
        """Test diffusion layer forward pass."""
        diffusion = DiffusionLayer(
            node_dim=64,
            hidden_dim=128,
            num_timesteps=10
        )
        
        # Test input
        batch_size = 4
        num_nodes = 20
        x_start = torch.randn(batch_size, num_nodes, 64)
        
        # Forward pass
        x_noisy, predicted_noise = diffusion(x_start)
        
        assert x_noisy.shape == x_start.shape
        assert predicted_noise.shape == x_start.shape
        
    def test_diffusion_sampling(self):
        """Test diffusion sampling process."""
        diffusion = DiffusionLayer(
            node_dim=32,
            hidden_dim=64,
            num_timesteps=5
        )
        
        # Sample from model
        shape = (2, 10, 32)
        device = torch.device("cpu")
        
        samples = diffusion.sample(shape, device, num_inference_steps=3)
        
        assert samples.shape == shape


class TestGraphLayers:
    """Test graph neural network layers."""
    
    def test_graph_convolution(self):
        """Test basic graph convolution."""
        conv = GraphConvolution(in_channels=32, out_channels=64)
        
        # Create test graph
        num_nodes = 10
        x = torch.randn(num_nodes, 32)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        out = conv(x, edge_index)
        
        assert out.shape == (num_nodes, 64)
        
    def test_dynamic_graph_layer(self):
        """Test dynamic graph layer."""
        layer = DynamicGraphLayer(
            node_dim=32,
            edge_dim=16,
            hidden_dim=64
        )
        
        # Create test data
        num_nodes = 15
        num_edges = 30
        x = torch.randn(num_nodes, 32)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 16)
        
        # Forward pass
        out = layer(x, edge_index, edge_attr)
        
        assert out.shape == (num_nodes, 32)  # Same as input due to residual


class TestAttention:
    """Test attention mechanisms."""
    
    def test_multi_head_attention(self):
        """Test multi-head attention."""
        attention = MultiHeadAttention(embed_dim=64, num_heads=8)
        
        batch_size = 2
        seq_len = 20
        query = torch.randn(batch_size, seq_len, 64)
        
        # Self-attention
        output, weights = attention(query)
        
        assert output.shape == query.shape
        if weights is not None:
            assert weights.shape == (batch_size, seq_len, seq_len)


class TestDGDMModel:
    """Test main DGDM model."""
    
    def create_test_graph(self, num_nodes=20, node_features=64):
        """Create a test graph."""
        x = torch.randn(num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        pos = torch.rand(num_nodes, 2)
        
        return Data(x=x, edge_index=edge_index, pos=pos, num_nodes=num_nodes)
        
    def test_model_creation(self):
        """Test DGDM model creation."""
        model = DGDMModel(
            node_features=64,
            hidden_dims=[128, 64],
            num_diffusion_steps=5,
            attention_heads=4,
            num_classes=3
        )
        
        assert model.node_features == 64
        assert model.hidden_dims == [128, 64]
        assert model.num_classes == 3
        
    def test_model_forward_inference(self):
        """Test model forward pass in inference mode."""
        model = DGDMModel(
            node_features=64,
            hidden_dims=[32, 16],
            num_diffusion_steps=3,
            num_classes=2
        )
        
        graph = self.create_test_graph(num_nodes=10, node_features=64)
        
        # Forward pass
        outputs = model(graph, mode="inference")
        
        assert "graph_embedding" in outputs
        assert "classification_logits" in outputs
        assert "classification_probs" in outputs
        
        # Check shapes
        assert outputs["graph_embedding"].shape[1] == 16  # Last hidden dim
        assert outputs["classification_logits"].shape[1] == 2  # Num classes
        
    def test_model_forward_pretrain(self):
        """Test model forward pass in pretrain mode."""
        model = DGDMModel(
            node_features=32,
            hidden_dims=[64, 32],
            num_diffusion_steps=3
        )
        
        graph = self.create_test_graph(num_nodes=15, node_features=32)
        
        # Pretraining step
        outputs = model.pretrain_step(graph, mask_ratio=0.2)
        
        assert "diffusion_loss" in outputs
        assert "total_pretrain_loss" in outputs
        
        # Check loss is tensor
        assert isinstance(outputs["diffusion_loss"], torch.Tensor)
        assert outputs["diffusion_loss"].numel() == 1


class TestLossFunctions:
    """Test loss functions."""
    
    def test_diffusion_loss(self):
        """Test diffusion loss computation."""
        loss_fn = DiffusionLoss()
        
        predicted_noise = torch.randn(10, 20, 64)
        target_noise = torch.randn(10, 20, 64)
        
        loss = loss_fn(predicted_noise, target_noise)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1
        assert loss.item() >= 0
        
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.1)
        
        embeddings = torch.randn(50, 128)
        batch_indices = torch.randint(0, 5, (50,))
        
        loss = loss_fn(embeddings, batch_indices)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.numel() == 1


class TestDataStructures:
    """Test data structures and utilities."""
    
    def test_graph_data_creation(self):
        """Test PyTorch Geometric Data creation."""
        num_nodes = 25
        node_features = 128
        
        x = torch.randn(num_nodes, node_features)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        
        assert data.num_nodes == num_nodes
        assert data.x.shape == (num_nodes, node_features)
        assert data.edge_index.shape[0] == 2
        
    def test_batch_processing(self):
        """Test batching of graphs."""
        from torch_geometric.data import Batch
        
        # Create multiple graphs
        graphs = []
        for i in range(3):
            num_nodes = 10 + i * 5
            x = torch.randn(num_nodes, 64)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes))
            graphs.append(Data(x=x, edge_index=edge_index, num_nodes=num_nodes))
            
        # Create batch
        batch = Batch.from_data_list(graphs)
        
        assert batch.batch is not None
        assert batch.num_graphs == 3
        assert batch.x.shape[0] == sum(g.num_nodes for g in graphs)


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])