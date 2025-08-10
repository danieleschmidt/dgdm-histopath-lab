"""
DGDM Histopath Lab: Dynamic Graph Diffusion Models for Whole-Slide Histopathology Analysis

A comprehensive framework for analyzing gigapixel whole-slide images using
state-of-the-art dynamic graph diffusion models with self-supervised learning.

Features:
- Dynamic Graph Diffusion Models with self-supervised learning
- Quantum-enhanced processing for large-scale analysis  
- Multi-scale tissue analysis with attention mechanisms
- Clinical-grade preprocessing and validation pipeline
- Production-ready deployment with monitoring and scaling
"""

__version__ = "0.1.0"
__author__ = "DGDM Team"
__email__ = "contact@example.com"
__license__ = "Apache-2.0"

# Build information
BUILD_INFO = {
    "version": __version__,
    "build": "quantum_enhanced_production",
    "features": [
        "Dynamic Graph Diffusion Models",
        "Self-Supervised Learning",
        "Quantum Enhancement",
        "Multi-Scale Analysis", 
        "Clinical Pipeline",
        "Production Deployment"
    ]
}

# Import core components with error handling
try:
    from dgdm_histopath.models.dgdm_model import DGDMModel
    from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
    from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder
    from dgdm_histopath.training.trainer import DGDMTrainer
    from dgdm_histopath.evaluation.predictor import DGDMPredictor
    from dgdm_histopath.evaluation.visualizer import AttentionVisualizer
    from dgdm_histopath.data.datamodule import HistopathDataModule
    
    CORE_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Core components not fully available: {e}")
    CORE_AVAILABLE = False

# Import quantum components if available
QUANTUM_AVAILABLE = False
try:
    from dgdm_histopath.quantum.quantum_planner import QuantumPlanner
    from dgdm_histopath.quantum.quantum_scheduler import QuantumScheduler
    QUANTUM_AVAILABLE = True
except ImportError:
    pass

def get_build_info():
    """Get build information."""
    return BUILD_INFO.copy()

def check_installation():
    """Check installation status."""
    return {
        "core_available": CORE_AVAILABLE,
        "quantum_available": QUANTUM_AVAILABLE,
        "version": __version__
    }

__all__ = [
    "DGDMModel",
    "SlideProcessor", 
    "TissueGraphBuilder",
    "DGDMTrainer",
    "DGDMPredictor",
    "AttentionVisualizer",
    "HistopathDataModule",
    "get_build_info",
    "check_installation",
    "BUILD_INFO",
    "CORE_AVAILABLE",
    "QUANTUM_AVAILABLE"
]