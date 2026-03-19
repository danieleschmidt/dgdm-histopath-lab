"""Tests for PatchGraph."""

import numpy as np
import torch
import pytest
from dgdm_histopath.graph import PatchGraph, _patch_features


class TestPatchFeatures:
    def test_shape(self):
        patch = np.zeros((32, 32), dtype=np.uint8)
        feats = _patch_features(patch)
        assert feats.shape == (4,)

    def test_uniform_patch(self):
        patch = np.full((32, 32), 128, dtype=np.uint8)
        feats = _patch_features(patch)
        assert abs(feats[0] - 128) < 1e-3   # mean ≈ 128
        assert feats[1] < 1e-3              # std ≈ 0
        assert feats[2] < 1e-3              # h_diff ≈ 0
        assert feats[3] < 1e-3              # contrast ≈ 0

    def test_high_contrast(self):
        patch = np.zeros((32, 32), dtype=np.uint8)
        patch[:, 16:] = 255
        feats = _patch_features(patch)
        assert feats[3] == 255.0            # contrast = max - min = 255


class TestPatchGraph:
    def _sample_image(self, H=64, W=64):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(H, W), dtype=np.uint8)

    def test_basic_build(self):
        pg = PatchGraph(patch_size=32, stride=32, connectivity=4)
        img = self._sample_image(64, 64)
        x, ei = pg.build(img)
        # 64×64 image, 32-px non-overlapping → 2×2 = 4 nodes
        assert x.shape == (4, 4)
        assert ei.shape[0] == 2
        assert ei.dtype == torch.long

    def test_node_count(self):
        pg = PatchGraph(patch_size=16, stride=16)
        img = self._sample_image(64, 64)
        x, _ = pg.build(img)
        assert x.shape[0] == 16   # 4×4 grid

    def test_edge_direction_undirected(self):
        """For a 4-connected grid all edges should appear in both directions."""
        pg = PatchGraph(patch_size=32, stride=32, connectivity=4)
        img = self._sample_image(64, 64)
        _, ei = pg.build(img)
        edges = set(zip(ei[0].tolist(), ei[1].tolist()))
        for u, v in list(edges):
            assert (v, u) in edges, f"Missing reverse edge ({v}, {u})"

    def test_8_connectivity_more_edges(self):
        pg4 = PatchGraph(patch_size=16, stride=16, connectivity=4)
        pg8 = PatchGraph(patch_size=16, stride=16, connectivity=8)
        img = self._sample_image(64, 64)
        _, ei4 = pg4.build(img)
        _, ei8 = pg8.build(img)
        assert ei8.size(1) > ei4.size(1)

    def test_node_features_dtype(self):
        pg = PatchGraph(patch_size=32, stride=32)
        img = self._sample_image()
        x, _ = pg.build(img)
        assert x.dtype == torch.float32

    def test_rgb_image(self):
        """3-channel image should be averaged to grayscale."""
        pg = PatchGraph(patch_size=32, stride=32)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        x, _ = pg.build(img)
        assert x.shape[0] > 0

    def test_small_image_raises(self):
        pg = PatchGraph(patch_size=32, stride=32)
        img = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError):
            pg.build(img)

    def test_grid_shape(self):
        pg = PatchGraph(patch_size=16, stride=8)
        img = self._sample_image(64, 64)
        r, c = pg.grid_shape(img)
        # (64 - 16) / 8 + 1 = 7 in each dim
        assert r == 7
        assert c == 7
