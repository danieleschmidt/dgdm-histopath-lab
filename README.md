# dgdm-histopath-lab

**Dynamic Graph Diffusion Model for Histopathology Image Analysis**

Classifies tissue patches as **NORMAL** or **TUMOR** by building a spatial graph from image sub-patches and running attention-weighted graph diffusion over it.

No `torch_geometric` required — message passing is implemented from scratch with pure PyTorch sparse operations.

---

## The Idea

Whole-slide histopathology images are too large to feed into a CNN directly.  
The standard approach: tile the slide into patches, then reason about *spatial relationships* between them.

This repo models those relationships as a **graph**:

```
Patch grid (2×2 example)     →    Spatial graph
┌──────┬──────┐                    0 ── 1
│  P0  │  P1  │                    │    │
├──────┼──────┤                    2 ── 3
│  P2  │  P3  │
└──────┴──────┘
```

Each node carries patch statistics:  
`[mean_intensity, std_intensity, texture_proxy, local_contrast]`

A 3-layer **GraphDiffusionLayer** (attention-weighted message passing) propagates context across the graph, then mean-pooling + MLP gives the slide-level classification.

---

## Architecture

### `PatchGraph`
Converts a grayscale image into a spatial graph.
- Nodes: non-overlapping patches (default 32×32 px)
- Node features: `[mean, std, h_diff, contrast]` — 4-dimensional
- Edges: 8-connected grid adjacency (or 4-connected)

### `GraphDiffusionLayer`
Message-passing layer with attention-weighted aggregation:
1. Project all node features: `h = W_msg · x`
2. Attention score per edge: `score = LeakyReLU(a_src[h_src] + a_dst[h_dst])`
3. Softmax over each node's neighbourhood
4. Weighted sum of neighbour messages
5. Residual connection + LayerNorm + ELU

### `HistoGNN`
Three stacked `GraphDiffusionLayer`s:
```
in_dim → 64 → 128 → 64 → mean-pool → Linear(64→32) → Linear(32→2)
```
36 k parameters total.

### `SyntheticHistoDataset`
Generates synthetic H&E-stained grayscale patches:
- **NORMAL**: uniform tissue, base intensity ~180, mild Gaussian texture
- **TUMOR**: darker base (~130), high-contrast nuclei blobs, irregular texture

---

## Quick Start

```bash
# Clone
git clone https://github.com/danieleschmidt/dgdm-histopath-lab
cd dgdm-histopath-lab

# Install (PyTorch required)
pip install -e .

# Run demo: train + classify 20 synthetic patches
python demo.py
```

Expected output:
```
Dataset: 20 patches  (NORMAL=10, TUMOR=10)
Graph per patch: 4 nodes, 12 edges  (2×2 grid)
Model: HistoGNN  (36,450 parameters)
Training on 16 patches for 80 epochs …
  epoch  20  loss=0.0010
...
Train accuracy : 100.0%
Test  accuracy : 100.0%  (4 held-out patches)
```

---

## Run Tests

```bash
pytest tests/ -v
# 31 tests, ~1.5 s
```

---

## Using with Real Data

```python
import numpy as np
from PIL import Image
from dgdm_histopath import PatchGraph, HistoGNN

# Load a grayscale WSI tile
img = np.array(Image.open("tile.png").convert("L"))

# Build graph
pg = PatchGraph(patch_size=64, stride=64, connectivity=8)
x, edge_index = pg.build(img)

# Classify
model = HistoGNN(in_dim=4)
logits, probs = model(x, edge_index)
print("NORMAL" if probs[0, 0] > 0.5 else "TUMOR")
```

For real slides, replace the 4-dim patch statistics with CNN-extracted features (e.g. ResNet-50 patch embeddings) and set `in_dim` accordingly.

---

## Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.0
- NumPy
- SciPy (optional, for future texture features)

No `torch_geometric` or OpenCV required.

---

## License

MIT
