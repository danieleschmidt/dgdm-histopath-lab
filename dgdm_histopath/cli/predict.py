"""
Prediction CLI for DGDM Histopath Lab.

Command-line interface for making predictions with trained DGDM models.
"""

import typer
import logging
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from dgdm_histopath.training.trainer import DGDMTrainer
from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder
from dgdm_histopath.evaluation.visualizer import AttentionVisualizer
from dgdm_histopath.utils.logging import setup_logging

app = typer.Typer(help="Make predictions with trained DGDM models")


@app.command()
def predict(
    model_path: str = typer.Option(..., help="Path to trained model checkpoint"),
    input_path: str = typer.Option(..., help="Path to input data (slide/graph/directory)"),
    output_dir: str = typer.Option("./predictions", help="Output directory for results"),
    
    # Input data type
    input_type: str = typer.Option("slide", help="Input type (slide/graph/directory)"),
    
    # Processing parameters
    patch_size: int = typer.Option(256, help="Patch size for slide processing"),
    magnifications: str = typer.Option("20.0", help="Magnifications (comma-separated)"),
    tissue_threshold: float = typer.Option(0.8, help="Tissue threshold"),
    max_patches: int = typer.Option(1000, help="Maximum patches per slide"),
    
    # Prediction parameters
    batch_size: int = typer.Option(4, help="Batch size for inference"),
    device: str = typer.Option("auto", help="Device (auto/cpu/cuda)"),
    
    # Output options
    save_embeddings: bool = typer.Option(True, help="Save graph embeddings"),
    save_attention: bool = typer.Option(True, help="Save attention weights"),
    save_visualizations: bool = typer.Option(False, help="Save attention visualizations"),
    output_format: str = typer.Option("json", help="Output format (json/csv/h5)"),
    
    # Misc
    debug: bool = typer.Option(False, help="Enable debug logging")
):
    """Make predictions on histopathology data using trained DGDM model."""
    
    # Setup logging
    setup_logging(level="DEBUG" if debug else "INFO")
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting DGDM prediction")
    logger.info(f"Model: {model_path}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading trained model...")
    try:
        model = DGDMTrainer.load_from_checkpoint(model_path)
        model.eval()
        model = model.to(device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise typer.Exit(1)
    
    # Parse magnifications
    magnifications_list = [float(x.strip()) for x in magnifications.split(",")]
    
    # Setup processors
    slide_processor = SlideProcessor(
        patch_size=patch_size,
        tissue_threshold=tissue_threshold,
        save_patches=False
    )
    
    graph_builder = TissueGraphBuilder(
        feature_extractor="dinov2"
    )
    
    # Setup visualizer if needed
    visualizer = None
    if save_visualizations:
        visualizer = AttentionVisualizer()
    
    # Process input
    input_path = Path(input_path)
    predictions = []
    
    if input_type == "slide" and input_path.is_file():
        # Single slide
        predictions = [_process_single_slide(
            input_path, model, slide_processor, graph_builder, 
            magnifications_list, max_patches, device, save_attention, logger
        )]
        
    elif input_type == "directory" and input_path.is_dir():
        # Multiple slides
        slide_files = []
        for ext in ["*.svs", "*.tiff", "*.ndpi"]:
            slide_files.extend(input_path.glob(ext))
            
        logger.info(f"Found {len(slide_files)} slides to process")
        
        for slide_file in tqdm(slide_files, desc="Processing slides"):
            try:
                pred = _process_single_slide(
                    slide_file, model, slide_processor, graph_builder,
                    magnifications_list, max_patches, device, save_attention, logger
                )
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Failed to process {slide_file}: {e}")
                continue
                
    elif input_type == "graph":
        # Pre-computed graph
        logger.info("Loading pre-computed graph...")
        if input_path.suffix == ".pt":
            graph = torch.load(input_path)
        else:
            raise ValueError(f"Unsupported graph format: {input_path.suffix}")
            
        pred = _predict_from_graph(graph, model, device, save_attention)
        pred["slide_id"] = input_path.stem
        predictions = [pred]
        
    else:
        logger.error(f"Invalid input type or path: {input_type}, {input_path}")
        raise typer.Exit(1)
    
    # Save predictions
    logger.info(f"Saving predictions for {len(predictions)} samples...")
    _save_predictions(predictions, output_path, output_format, save_embeddings)
    
    # Generate visualizations if requested
    if save_visualizations and visualizer is not None:
        logger.info("Generating attention visualizations...")
        _generate_visualizations(predictions, visualizer, output_path)
    
    logger.info("Prediction completed successfully!")


def _process_single_slide(
    slide_path: Path,
    model,
    slide_processor: SlideProcessor,
    graph_builder: TissueGraphBuilder,
    magnifications: List[float],
    max_patches: int,
    device: str,
    save_attention: bool,
    logger
) -> dict:
    """Process a single slide and make prediction."""
    
    slide_id = slide_path.stem
    logger.info(f"Processing slide: {slide_id}")
    
    # Process slide
    slide_data = slide_processor.process_slide(
        slide_path, magnifications, max_patches
    )
    
    # Build graph
    graph = graph_builder.build_graph(slide_data)
    graph = graph.to(device)
    
    # Make prediction
    prediction = _predict_from_graph(graph, model, device, save_attention)
    prediction["slide_id"] = slide_id
    prediction["slide_path"] = str(slide_path)
    prediction["num_patches"] = len(slide_data.patches)
    
    return prediction


def _predict_from_graph(graph, model, device: str, save_attention: bool) -> dict:
    """Make prediction from graph data."""
    
    with torch.no_grad():
        # Forward pass
        outputs = model.forward(
            graph, mode="inference", 
            return_attention=save_attention, 
            return_embeddings=True
        )
        
        prediction = {
            "graph_embedding": outputs["graph_embedding"].cpu().numpy(),
        }
        
        # Add task-specific predictions
        if "classification_probs" in outputs:
            probs = outputs["classification_probs"].cpu().numpy()
            prediction["classification_probs"] = probs
            prediction["predicted_class"] = int(np.argmax(probs))
            prediction["confidence"] = float(np.max(probs))
            
        if "regression_outputs" in outputs:
            prediction["regression_outputs"] = outputs["regression_outputs"].cpu().numpy()
            
        if save_attention and "attention_weights" in outputs:
            prediction["attention_weights"] = outputs["attention_weights"].cpu().numpy()
            
        if "node_embeddings" in outputs:
            prediction["node_embeddings"] = outputs["node_embeddings"].cpu().numpy()
            
    return prediction


def _save_predictions(predictions: List[dict], output_path: Path, format: str, save_embeddings: bool):
    """Save predictions to file."""
    
    if format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_predictions = []
        for pred in predictions:
            json_pred = {}
            for key, value in pred.items():
                if isinstance(value, np.ndarray):
                    if save_embeddings or "embedding" not in key.lower():
                        json_pred[key] = value.tolist()
                else:
                    json_pred[key] = value
            json_predictions.append(json_pred)
            
        with open(output_path / "predictions.json", "w") as f:
            json.dump(json_predictions, f, indent=2)
            
    elif format == "csv":
        # Create summary CSV
        summary_data = []
        for pred in predictions:
            row = {
                "slide_id": pred.get("slide_id", "unknown"),
                "predicted_class": pred.get("predicted_class"),
                "confidence": pred.get("confidence"),
                "num_patches": pred.get("num_patches")
            }
            
            # Add regression outputs if available
            if "regression_outputs" in pred:
                reg_outputs = pred["regression_outputs"]
                for i, val in enumerate(reg_outputs):
                    row[f"regression_output_{i}"] = val
                    
            summary_data.append(row)
            
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path / "predictions.csv", index=False)
        
        # Save embeddings separately if requested
        if save_embeddings:
            embeddings = np.stack([pred["graph_embedding"] for pred in predictions])
            np.save(output_path / "embeddings.npy", embeddings)
            
    elif format == "h5":
        import h5py
        
        with h5py.File(output_path / "predictions.h5", "w") as f:
            for i, pred in enumerate(predictions):
                group = f.create_group(f"prediction_{i}")
                
                for key, value in pred.items():
                    if isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value)
                    elif isinstance(value, (int, float, str)):
                        group.attrs[key] = value
                        
    else:
        raise ValueError(f"Unsupported output format: {format}")


def _generate_visualizations(predictions: List[dict], visualizer, output_path: Path):
    """Generate attention visualizations."""
    
    viz_dir = output_path / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    for pred in predictions:
        if "attention_weights" in pred:
            slide_id = pred["slide_id"]
            
            # Create attention heatmap (simplified)
            attention = pred["attention_weights"]
            
            # Save attention map
            np.save(viz_dir / f"{slide_id}_attention.npy", attention)


@app.command()
def batch_predict(
    model_path: str = typer.Option(..., help="Path to trained model checkpoint"),
    input_csv: str = typer.Option(..., help="CSV file with slide paths and metadata"),
    output_dir: str = typer.Option("./batch_predictions", help="Output directory"),
    batch_size: int = typer.Option(4, help="Batch size for processing"),
    num_workers: int = typer.Option(4, help="Number of parallel workers")
):
    """Run batch prediction on multiple slides from CSV file."""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load CSV
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} slides from CSV")
    
    # Process in batches
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # This would implement parallel processing
    logger.info("Batch prediction not yet implemented")
    

if __name__ == "__main__":
    app()