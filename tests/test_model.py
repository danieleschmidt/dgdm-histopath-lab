"""Tests for GraphDiffusionLayer and HistoGNN."""

import torch
import pytest
from dgdm_histopath.model import GraphDiffusionLayer, HistoGNN, _scatter_softmax, _scatter_sum


class TestScatterHelpers:
    def test_scatter_softmax_sum_to_one(self):
        """Each destination node's weights should sum to 1."""
        scores = torch.tensor([1.0, 2.0, 0.5, -1.0, 3.0])
        dst    = torch.tensor([0,   0,   1,    1,   2])
        w = _scatter_softmax(scores, dst, num_nodes=3)
        assert w.shape == scores.shape
        # Weights for node 0: indices 0,1
        assert abs(w[0].item() + w[1].item() - 1.0) < 1e-5
        # Weights for node 1: indices 2,3
        assert abs(w[2].item() + w[3].item() - 1.0) < 1e-5

    def test_scatter_sum_shape(self):
        values = torch.randn(5, 8)
        dst    = torch.tensor([0, 1, 2, 0, 1])
        out = _scatter_sum(values, dst, num_nodes=3)
        assert out.shape == (3, 8)


class TestGraphDiffusionLayer:
    def _small_graph(self):
        N, D = 6, 4
        x = torch.randn(N, D)
        # Simple chain: 0-1-2-3-4-5 (undirected)
        src = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
        dst = [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
        ei = torch.tensor([src, dst], dtype=torch.long)
        return x, ei

    def test_output_shape(self):
        layer = GraphDiffusionLayer(in_dim=4, out_dim=16)
        x, ei = self._small_graph()
        out = layer(x, ei)
        assert out.shape == (6, 16)

    def test_same_dim_no_projection(self):
        layer = GraphDiffusionLayer(in_dim=8, out_dim=8)
        x = torch.randn(5, 8)
        ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        out = layer(x, ei)
        assert out.shape == (5, 8)

    def test_gradient_flows(self):
        layer = GraphDiffusionLayer(in_dim=4, out_dim=8)
        x, ei = self._small_graph()
        x.requires_grad_(True)
        out = layer(x, ei)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not x.grad.isnan().any()

    def test_no_isolated_node_problem(self):
        """Node with no edges should still produce a valid output (zero agg + residual)."""
        layer = GraphDiffusionLayer(in_dim=4, out_dim=4)
        x = torch.randn(3, 4)
        # Only node 0-1 connected; node 2 is isolated
        ei = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = layer(x, ei)
        assert not out.isnan().any()


class TestHistoGNN:
    def _make_graph(self, n_nodes=9, in_dim=4):
        x = torch.randn(n_nodes, in_dim)
        # 3×3 grid edges
        edges = []
        for i in range(n_nodes):
            r, c = divmod(i, 3)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    j = nr * 3 + nc
                    edges.append((i, j))
        ei = torch.tensor(edges, dtype=torch.long).T
        return x, ei

    def test_output_shape(self):
        model = HistoGNN(in_dim=4, hidden=32, num_classes=2)
        x, ei = self._make_graph()
        logits, probs = model(x, ei)
        assert logits.shape == (1, 2)
        assert probs.shape  == (1, 2)

    def test_probs_sum_to_one(self):
        model = HistoGNN(in_dim=4, hidden=16)
        x, ei = self._make_graph()
        _, probs = model(x, ei)
        assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_train_loss_decreases(self):
        """Single graph: loss should decrease over 30 gradient steps."""
        import torch.nn as nn
        model = HistoGNN(in_dim=4, hidden=32)
        x, ei = self._make_graph()
        label = torch.tensor([1], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)

        losses = []
        for _ in range(30):
            logits, _ = model(x, ei)
            loss = criterion(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should be lower at end than start
        assert losses[-1] < losses[0], "Loss did not decrease during training"

    def test_parameter_count(self):
        model = HistoGNN(in_dim=4, hidden=64)
        n = sum(p.numel() for p in model.parameters())
        # Sanity: should be a few thousand, not millions
        assert 1_000 < n < 100_000
