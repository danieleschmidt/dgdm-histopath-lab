"""
PatchGraph: builds a spatial graph from image patches.

Nodes  = tissue patches (each patch → node feature vector)
Edges  = spatially adjacent patches (8-connectivity grid neighbourhood)
Node features:
  - mean pixel intensity
  - std pixel intensity
  - GLCM-like texture proxy (mean absolute difference between adjacent pixels)
  - local contrast (max - min in patch)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple


def _patch_features(patch: np.ndarray) -> np.ndarray:
    """Return a 4-dim feature vector for a single grayscale patch (H×W)."""
    flat = patch.astype(np.float32).ravel()
    mean = flat.mean()
    std = flat.std()
    # Horizontal absolute differences (texture proxy)
    h_diff = np.abs(np.diff(patch.astype(np.float32), axis=1)).mean()
    contrast = float(flat.max() - flat.min())
    return np.array([mean, std, h_diff, contrast], dtype=np.float32)


class PatchGraph:
    """
    Converts a 2-D grid of image patches into a graph (node features + edge index).

    Parameters
    ----------
    patch_size : int
        Side length (pixels) of each square patch.
    stride : int, optional
        Stride between patches. Defaults to patch_size (non-overlapping).
    connectivity : int
        4  → 4-connected (up/down/left/right)
        8  → 8-connected (+ diagonals)  [default]
    """

    def __init__(self, patch_size: int = 32, stride: int | None = None, connectivity: int = 8):
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.connectivity = connectivity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build graph from a 2-D (H×W) or 3-D (H×W×C) grayscale image.

        Returns
        -------
        node_features : torch.Tensor  shape (N, 4)
        edge_index    : torch.Tensor  shape (2, E)   COO format, undirected
        """
        if image.ndim == 3:
            # Convert to grayscale by averaging channels
            image = image.mean(axis=-1)

        H, W = image.shape
        ps, st = self.patch_size, self.stride

        rows = list(range(0, H - ps + 1, st))
        cols = list(range(0, W - ps + 1, st))
        n_rows, n_cols = len(rows), len(cols)

        if n_rows == 0 or n_cols == 0:
            raise ValueError(
                f"Image ({H}×{W}) too small for patch_size={ps}. "
                "Use a larger image or smaller patch_size."
            )

        # ---- Node features --------------------------------------------------
        features = []
        for r in rows:
            for c in cols:
                patch = image[r : r + ps, c : c + ps]
                features.append(_patch_features(patch))

        node_features = torch.tensor(np.stack(features), dtype=torch.float32)

        # ---- Edge index (grid adjacency) ------------------------------------
        def node_id(ri, ci):
            return ri * n_cols + ci

        src, dst = [], []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connectivity == 8:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for ri in range(n_rows):
            for ci in range(n_cols):
                u = node_id(ri, ci)
                for dr, dc in directions:
                    nr, nc = ri + dr, ci + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        v = node_id(nr, nc)
                        src.append(u)
                        dst.append(v)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return node_features, edge_index

    def grid_shape(self, image: np.ndarray) -> Tuple[int, int]:
        """Return (n_rows, n_cols) patch grid for a given image."""
        H, W = image.shape[:2]
        ps, st = self.patch_size, self.stride
        return len(range(0, H - ps + 1, st)), len(range(0, W - ps + 1, st))
