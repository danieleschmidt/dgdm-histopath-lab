"""
DGDM model predictor for inference and evaluation.

Provides high-level interface for making predictions with trained DGDM models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import logging

from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder
from torch_geometric.data import Data


class DGDMPredictor:
    """
    High-level predictor interface for trained DGDM models.
    
    Handles model loading, preprocessing, and prediction generation
    with support for various input formats and output options.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        preprocessing_config: Optional[Dict] = None
    ):
        """
        Initialize DGDM predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device for inference ("auto", "cpu", "cuda")
            preprocessing_config: Configuration for preprocessing pipeline
        """
        self.model_path = Path(model_path)
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
        
        # Setup preprocessing pipeline
        self._setup_preprocessing(preprocessing_config or {})
        
    def _load_model(self):
        """Load trained model from checkpoint."""
        
        self.logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.model = DGDMTrainer.load_from_checkpoint(self.model_path)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            self.logger.info("Model loaded successfully")
            
            # Extract model configuration
            self.model_config = self.model.hparams
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def _setup_preprocessing(self, config: Dict):
        """Setup preprocessing pipeline."""
        
        # Default preprocessing configuration
        default_config = {
            "patch_size": 256,
            "magnifications": [20.0],
            "tissue_threshold": 0.8,
            "max_patches": 1000,
            "feature_extractor": "dinov2"
        }
        
        # Update with provided config
        preprocessing_config = {**default_config, **config}
        
        # Initialize processors
        self.slide_processor = SlideProcessor(
            patch_size=preprocessing_config["patch_size"],
            tissue_threshold=preprocessing_config["tissue_threshold"],
            save_patches=False
        )
        
        self.graph_builder = TissueGraphBuilder(
            feature_extractor=preprocessing_config["feature_extractor"]
        )
        
        self.preprocessing_config = preprocessing_config
        
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = "auto",
        **kwargs
    ) -> 'DGDMPredictor':
        """
        Create predictor from model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference
            **kwargs: Additional arguments
            
        Returns:
            Initialized predictor
        """
        return cls(checkpoint_path, device=device, **kwargs)
        
    def predict_slide(
        self,
        slide_path: Union[str, Path],
        return_attention: bool = False,
        return_embeddings: bool = False,
        return_patches: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on a whole-slide image.
        
        Args:
            slide_path: Path to slide file
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return embeddings
            return_patches: Whether to return patch information
            
        Returns:
            Dictionary containing predictions and optional outputs
        """
        
        slide_path = Path(slide_path)
        slide_id = slide_path.stem
        
        self.logger.info(f"Processing slide: {slide_id}")
        
        # Process slide to extract patches
        slide_data = self.slide_processor.process_slide(
            slide_path,
            self.preprocessing_config["magnifications"],
            self.preprocessing_config["max_patches"]
        )
        
        # Build tissue graph
        graph = self.graph_builder.build_graph(slide_data)
        
        # Make prediction on graph
        prediction = self.predict_graph(
            graph,
            return_attention=return_attention,
            return_embeddings=return_embeddings
        )
        
        # Add slide metadata
        prediction["slide_id"] = slide_id
        prediction["slide_path"] = str(slide_path) 
        prediction["num_patches"] = len(slide_data.patches)
        prediction["slide_metadata"] = slide_data.metadata
        
        # Add patch information if requested
        if return_patches:
            prediction["patches"] = [
                {
                    "patch_id": patch.patch_id,
                    "x": patch.x,
                    "y": patch.y,
                    "tissue_percentage": patch.tissue_percentage,
                    "magnification": patch.magnification
                }
                for patch in slide_data.patches
            ]
            
        return prediction
        
    def predict_graph(
        self,
        graph: Data,
        return_attention: bool = False,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction on a tissue graph.
        
        Args:
            graph: Tissue graph data
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return embeddings
            
        Returns:
            Dictionary containing predictions
        """
        
        # Move graph to device
        graph = graph.to(self.device)
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model.forward(
                graph,
                mode="inference",
                return_attention=return_attention,
                return_embeddings=return_embeddings
            )
            
            # Extract predictions
            prediction = {}
            
            # Classification predictions
            if "classification_probs" in outputs:
                probs = outputs["classification_probs"].cpu().numpy()
                prediction["classification_probs"] = probs
                prediction["predicted_class"] = int(np.argmax(probs))
                prediction["confidence"] = float(np.max(probs))
                
                # Add class-wise probabilities
                for i, prob in enumerate(probs):
                    prediction[f"class_{i}_prob"] = float(prob)
                    
            # Regression predictions
            if "regression_outputs" in outputs:
                reg_outputs = outputs["regression_outputs"].cpu().numpy()
                prediction["regression_outputs"] = reg_outputs
                
                # Add individual regression targets
                for i, output in enumerate(reg_outputs):
                    prediction[f"regression_target_{i}"] = float(output)
                    
            # Graph embedding
            if "graph_embedding" in outputs:
                prediction["graph_embedding"] = outputs["graph_embedding"].cpu().numpy()
                
            # Node embeddings
            if return_embeddings and "node_embeddings" in outputs:
                prediction["node_embeddings"] = outputs["node_embeddings"].cpu().numpy()
                
            # Attention weights
            if return_attention and "attention_weights" in outputs:
                prediction["attention_weights"] = outputs["attention_weights"].cpu().numpy()
                
            # Graph statistics
            prediction["num_nodes"] = graph.num_nodes
            prediction["num_edges"] = graph.num_edges // 2  # Undirected graph
            
        return prediction
        
    def predict_batch(
        self,
        graphs: List[Data],
        batch_size: int = 8,
        return_attention: bool = False,
        return_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of graphs.
        
        Args:
            graphs: List of tissue graphs
            batch_size: Batch size for processing
            return_attention: Whether to return attention weights
            return_embeddings: Whether to return embeddings
            
        Returns:
            List of prediction dictionaries
        """
        
        predictions = []
        
        # Process in batches
        for i in range(0, len(graphs), batch_size):
            batch_graphs = graphs[i:i + batch_size]
            
            for graph in batch_graphs:
                pred = self.predict_graph(
                    graph,
                    return_attention=return_attention,
                    return_embeddings=return_embeddings
                )
                predictions.append(pred)
                
        return predictions
        
    def extract_biomarkers(
        self,
        prediction: Dict[str, Any],
        top_k: int = 10,
        method: str = "attention"
    ) -> Dict[str, Any]:
        """
        Extract interpretable biomarkers from predictions.
        
        Args:
            prediction: Prediction dictionary
            top_k: Number of top biomarkers to extract
            method: Method for biomarker extraction
            
        Returns:
            Dictionary containing biomarker information
        """
        
        biomarkers = {
            "method": method,
            "top_k": top_k,
            "biomarkers": []
        }
        
        if method == "attention" and "attention_weights" in prediction:
            # Use attention weights to identify important regions
            attention = prediction["attention_weights"]
            
            if attention.ndim == 2:  # [num_nodes, num_nodes]
                # Compute node importance as sum of incoming attention
                importance = attention.sum(axis=0)
            else:  # [num_nodes]
                importance = attention
                
            # Get top-k most important nodes
            top_indices = np.argsort(importance)[-top_k:][::-1]
            
            for i, idx in enumerate(top_indices):
                biomarkers["biomarkers"].append({
                    "rank": i + 1,
                    "node_index": int(idx),
                    "importance_score": float(importance[idx]),
                    "attention_weight": float(attention[idx] if attention.ndim == 1 else attention[idx, idx])
                })
                
        elif method == "embedding" and "node_embeddings" in prediction:
            # Use embedding magnitudes to identify important nodes
            embeddings = prediction["node_embeddings"]
            importance = np.linalg.norm(embeddings, axis=1)
            
            top_indices = np.argsort(importance)[-top_k:][::-1]
            
            for i, idx in enumerate(top_indices):
                biomarkers["biomarkers"].append({
                    "rank": i + 1,
                    "node_index": int(idx),
                    "importance_score": float(importance[idx]),
                    "embedding_norm": float(importance[idx])
                })
                
        return biomarkers
        
    def compute_uncertainty(
        self,
        prediction: Dict[str, Any],
        method: str = "entropy"
    ) -> Dict[str, float]:
        """
        Compute prediction uncertainty measures.
        
        Args:
            prediction: Prediction dictionary
            method: Uncertainty computation method
            
        Returns:
            Dictionary containing uncertainty measures
        """
        
        uncertainty = {"method": method}
        
        if "classification_probs" in prediction:
            probs = prediction["classification_probs"]
            
            if method == "entropy":
                # Shannon entropy
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                uncertainty["entropy"] = float(entropy)
                
            elif method == "max_prob":
                # 1 - max probability
                max_prob_uncertainty = 1.0 - np.max(probs)
                uncertainty["max_prob_uncertainty"] = float(max_prob_uncertainty)
                
            elif method == "margin":
                # Margin between top two predictions
                sorted_probs = np.sort(probs)[::-1]
                margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
                uncertainty["margin"] = float(1.0 - margin)
                
        return uncertainty
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        
        info = {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "model_config": dict(self.model_config) if hasattr(self, 'model_config') else {},
            "preprocessing_config": self.preprocessing_config,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        # Add task-specific information
        if hasattr(self.model.model, 'classification_head') and self.model.model.classification_head is not None:
            info["supports_classification"] = True
            info["num_classes"] = self.model.model.num_classes
        else:
            info["supports_classification"] = False
            
        if hasattr(self.model.model, 'regression_head') and self.model.model.regression_head is not None:
            info["supports_regression"] = True
            info["regression_targets"] = self.model.model.regression_targets
        else:
            info["supports_regression"] = False
            
        return info