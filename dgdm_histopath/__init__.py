"""
dgdm-histopath-lab
==================
Dynamic Graph Diffusion Model for histopathology image analysis.

Builds spatial graphs from tissue patches and classifies them using
attention-weighted graph diffusion — no torch_geometric required.
"""

from dgdm_histopath.graph import PatchGraph
from dgdm_histopath.model import GraphDiffusionLayer, HistoGNN
from dgdm_histopath.dataset import SyntheticHistoDataset

__version__ = "0.2.0"
__all__ = ["PatchGraph", "GraphDiffusionLayer", "HistoGNN", "SyntheticHistoDataset"]
