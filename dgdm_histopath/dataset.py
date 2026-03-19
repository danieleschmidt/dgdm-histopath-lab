"""
SyntheticHistoDataset: generates synthetic H&E-stained grayscale tissue patches.

Class patterns (grayscale, 0-255):
  NORMAL (0): relatively uniform, low-contrast tissue.
              Base intensity ~180, small Gaussian noise.
  TUMOR  (1): irregular, high-contrast regions.
              Multi-scale noise + dark "nuclei" blobs.

Each sample is a dict:
  {
    "image"      : np.ndarray (H×W) uint8
    "label"      : int  0=NORMAL, 1=TUMOR
    "label_name" : str
  }
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict


_LABEL_NAMES = {0: "NORMAL", 1: "TUMOR"}


def _make_normal_patch(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """Uniform tissue with mild Gaussian texture."""
    base = rng.normal(loc=180, scale=12, size=(size, size))
    # Subtle low-frequency variation (simulate stain gradient)
    x = np.linspace(-1, 1, size)
    gradient = np.outer(np.sin(x * 1.5), np.cos(x * 1.5)) * 8
    patch = base + gradient
    return np.clip(patch, 0, 255).astype(np.uint8)


def _make_tumor_patch(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    """Irregular, high-contrast tissue with dark nuclei."""
    # Darker, more variable base
    base = rng.normal(loc=130, scale=30, size=(size, size))

    # Add 4-8 "nuclei" (dark elliptical blobs)
    n_nuclei = rng.integers(4, 9)
    for _ in range(n_nuclei):
        cx = rng.integers(5, size - 5)
        cy = rng.integers(5, size - 5)
        rx = rng.integers(3, 8)
        ry = rng.integers(3, 8)
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx - cx) ** 2 / rx**2 + (yy - cy) ** 2 / ry**2) <= 1
        base[mask] -= rng.uniform(40, 80)

    # High-frequency noise (nuclear texture)
    base += rng.normal(0, 15, size=(size, size))

    return np.clip(base, 0, 255).astype(np.uint8)


class SyntheticHistoDataset:
    """
    Generates a list of synthetic tissue patches.

    Parameters
    ----------
    n_samples : int
        Total number of patches (split evenly between NORMAL and TUMOR).
    patch_size : int
        Spatial size of each square patch.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_samples: int = 20, patch_size: int = 64, seed: int = 42):
        self.n_samples = n_samples
        self.patch_size = patch_size
        self.seed = seed
        self._data: List[Dict] = []
        self._generate()

    def _generate(self) -> None:
        rng = np.random.default_rng(self.seed)
        n_normal = self.n_samples // 2
        n_tumor = self.n_samples - n_normal

        for _ in range(n_normal):
            self._data.append({
                "image": _make_normal_patch(rng, self.patch_size),
                "label": 0,
                "label_name": "NORMAL",
            })
        for _ in range(n_tumor):
            self._data.append({
                "image": _make_tumor_patch(rng, self.patch_size),
                "label": 1,
                "label_name": "TUMOR",
            })

        # Shuffle
        indices = rng.permutation(len(self._data)).tolist()
        self._data = [self._data[i] for i in indices]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict:
        return self._data[idx]

    def images(self) -> List[np.ndarray]:
        return [d["image"] for d in self._data]

    def labels(self) -> List[int]:
        return [d["label"] for d in self._data]
