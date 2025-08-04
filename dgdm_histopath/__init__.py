"""
DGDM Histopath Lab: Dynamic Graph Diffusion Models for Whole-Slide Histopathology Analysis

A comprehensive framework for analyzing gigapixel whole-slide images using
state-of-the-art dynamic graph diffusion models with self-supervised learning.
"""

__version__ = "0.1.0"
__author__ = "DGDM Team"
__email__ = "contact@example.com"
__license__ = "Apache-2.0"

from dgdm_histopath.models.dgdm_model import DGDMModel
from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder
from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.evaluation.predictor import DGDMPredictor
from dgdm_histopath.evaluation.visualizer import AttentionVisualizer
from dgdm_histopath.data.datamodule import HistopathDataModule

__all__ = [
    "DGDMModel",
    "SlideProcessor", 
    "TissueGraphBuilder",
    "DGDMTrainer",
    "DGDMPredictor",
    "AttentionVisualizer",
    "HistopathDataModule",
]