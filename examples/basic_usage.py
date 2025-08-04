#!/usr/bin/env python3
"""
Basic usage example for DGDM Histopath Lab.

This example demonstrates how to:
1. Load and preprocess a whole-slide image
2. Build a tissue graph representation
3. Train a DGDM model
4. Make predictions and visualize results
"""

import logging
from pathlib import Path
import torch
import numpy as np

# Import DGDM components
from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder
from dgdm_histopath.models.dgdm_model import DGDMModel
from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.evaluation.predictor import DGDMPredictor
from dgdm_histopath.evaluation.visualizer import AttentionVisualizer
from dgdm_histopath.utils.logging import setup_logging


def main():
    """Run basic DGDM usage example."""
    
    # Setup logging
    setup_logging(level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("Starting DGDM basic usage example")
    
    # Configuration
    config = {
        "patch_size": 256,
        "magnifications": [20.0],
        "tissue_threshold": 0.8,
        "max_patches": 500,  # Small number for example
        "feature_extractor": "dinov2"
    }
    
    # Example slide path (replace with actual path)
    slide_path = "path/to/example_slide.svs"
    output_dir = Path("./example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Check if slide exists
    if not Path(slide_path).exists():
        logger.warning(f"Example slide not found: {slide_path}")
        logger.info("Please provide a valid slide path to run this example")
        return
    
    # Step 1: Process slide and extract patches
    logger.info("Step 1: Processing slide and extracting patches")
    
    slide_processor = SlideProcessor(
        patch_size=config["patch_size"],
        tissue_threshold=config["tissue_threshold"],
        save_patches=False
    )
    
    try:
        slide_data = slide_processor.process_slide(
            slide_path,
            config["magnifications"],
            config["max_patches"]
        )
        
        logger.info(f"Extracted {len(slide_data.patches)} patches from slide")
        
    except Exception as e:
        logger.error(f"Failed to process slide: {e}")
        return
    
    # Step 2: Build tissue graph
    logger.info("Step 2: Building tissue graph representation")
    
    graph_builder = TissueGraphBuilder(
        feature_extractor=config["feature_extractor"]
    )
    
    try:
        tissue_graph = graph_builder.build_graph(slide_data)
        logger.info(f"Built tissue graph with {tissue_graph.num_nodes} nodes and {tissue_graph.num_edges} edges")
        
        # Save graph for later use
        torch.save(tissue_graph, output_dir / "example_graph.pt")
        
    except Exception as e:
        logger.error(f"Failed to build tissue graph: {e}")
        return
    
    # Step 3: Create and configure DGDM model
    logger.info("Step 3: Creating DGDM model")
    
    model = DGDMModel(
        node_features=768,  # DINOv2 feature dimension
        hidden_dims=[512, 256, 128],
        num_diffusion_steps=10,
        attention_heads=8,
        dropout=0.1,
        num_classes=2  # Binary classification example
    )
    
    logger.info(f"Created DGDM model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Step 4: Setup trainer (minimal example)
    logger.info("Step 4: Setting up trainer for minimal training example")
    
    trainer = DGDMTrainer(
        model=model,
        learning_rate=1e-4,
        pretrain_epochs=5,  # Very short for example
        finetune_epochs=5
    )
    
    # Create dummy data for demonstration
    dummy_graphs = [tissue_graph] * 10  # Replicate graph for batch
    
    logger.info("Running minimal training example (not a real training loop)")
    
    # This would normally be done with proper DataLoader
    # For demonstration only:
    try:
        # Simulate a training step
        model.train()
        outputs = model.pretrain_step(tissue_graph, mask_ratio=0.15)
        logger.info(f"Training step completed. Loss: {outputs['total_pretrain_loss'].item():.4f}")
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
    
    # Step 5: Make predictions
    logger.info("Step 5: Making predictions with the model")
    
    try:
        model.eval()
        with torch.no_grad():
            prediction = model.forward(tissue_graph, mode="inference", return_attention=True)
            
        logger.info("Prediction completed successfully")
        
        # Print prediction results
        if "classification_probs" in prediction:
            probs = prediction["classification_probs"].cpu().numpy()
            logger.info(f"Classification probabilities: {probs}")
            
        if "graph_embedding" in prediction:
            embedding = prediction["graph_embedding"].cpu().numpy()
            logger.info(f"Graph embedding shape: {embedding.shape}")
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return
    
    # Step 6: Visualize results
    logger.info("Step 6: Creating visualizations")
    
    try:
        visualizer = AttentionVisualizer()
        
        # Create prediction summary
        pred_dict = {
            "classification_probs": prediction["classification_probs"].cpu().numpy(),
            "graph_embedding": prediction["graph_embedding"].cpu().numpy()
        }
        
        if "attention_weights" in prediction:
            pred_dict["attention_weights"] = prediction["attention_weights"].cpu().numpy()
        
        # Generate visualization
        fig = visualizer.create_prediction_summary(
            pred_dict,
            title="DGDM Prediction Example",
            save_path=output_dir / "prediction_summary.png"
        )
        
        logger.info(f"Saved prediction visualization to {output_dir / 'prediction_summary.png'}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    
    # Step 7: Demonstrate predictor interface
    logger.info("Step 7: Demonstrating high-level predictor interface")
    
    try:
        # Save model checkpoint for predictor
        checkpoint_path = output_dir / "example_model.ckpt"
        trainer.save_model(checkpoint_path)
        
        # Create predictor
        predictor = DGDMPredictor(
            model_path=checkpoint_path,
            preprocessing_config=config
        )
        
        # Get model info
        model_info = predictor.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        logger.info("Predictor interface demonstrated successfully")
        
    except Exception as e:
        logger.error(f"Predictor demonstration failed: {e}")
    
    logger.info("DGDM basic usage example completed!")
    logger.info(f"Check output directory: {output_dir}")


def create_synthetic_example():
    """Create a synthetic example when no real slide is available."""
    
    logger = logging.getLogger(__name__)
    logger.info("Creating synthetic tissue graph for demonstration")
    
    # Create synthetic graph data
    num_nodes = 100
    node_features = torch.randn(num_nodes, 768)  # DINOv2 features
    
    # Create random edges (spatial connectivity)
    edge_prob = 0.1
    adj_matrix = torch.rand(num_nodes, num_nodes) < edge_prob
    adj_matrix = adj_matrix | adj_matrix.t()  # Make symmetric
    adj_matrix.fill_diagonal_(False)  # No self-loops
    
    edge_index = adj_matrix.nonzero().t()
    
    # Create positions
    positions = torch.rand(num_nodes, 2)
    
    # Create synthetic graph
    from torch_geometric.data import Data
    
    synthetic_graph = Data(
        x=node_features,
        edge_index=edge_index,
        pos=positions,
        num_nodes=num_nodes
    )
    
    logger.info(f"Created synthetic graph with {num_nodes} nodes and {edge_index.size(1)} edges")
    
    return synthetic_graph


if __name__ == "__main__":
    main()