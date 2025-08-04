"""
Preprocessing CLI for DGDM Histopath Lab.

Command-line interface for preprocessing whole-slide images and building tissue graphs.
"""

import typer
import logging
from pathlib import Path
from typing import Optional, List
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch

from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder
from dgdm_histopath.utils.logging import setup_logging

app = typer.Typer(help="Preprocess histopathology slides and build tissue graphs")


@app.command()
def process_slides(
    input_dir: str = typer.Option(..., help="Directory containing slide files"),
    output_dir: str = typer.Option(..., help="Output directory for processed data"),
    
    # Slide processing parameters
    patch_size: int = typer.Option(256, help="Size of patches to extract"),
    magnifications: str = typer.Option("20.0", help="Target magnifications (comma-separated)"),
    tissue_threshold: float = typer.Option(0.8, help="Minimum tissue percentage for patches"),
    overlap: float = typer.Option(0.0, help="Overlap between patches (0.0-1.0)"),
    max_patches: Optional[int] = typer.Option(None, help="Maximum patches per slide"),
    
    # Tissue detection parameters
    background_threshold: int = typer.Option(220, help="Background intensity threshold"),
    min_tissue_area: int = typer.Option(1000, help="Minimum tissue area in pixels"),
    
    # Processing options
    normalize_stains: bool = typer.Option(True, help="Apply stain normalization"),
    save_patches: bool = typer.Option(False, help="Save individual patches"),
    save_thumbnails: bool = typer.Option(True, help="Save slide thumbnails"),
    
    # Parallel processing
    num_workers: int = typer.Option(4, help="Number of parallel workers"),
    
    # Output options
    output_format: str = typer.Option("h5", help="Output format (h5/pt)"),
    
    # Misc
    debug: bool = typer.Option(False, help="Enable debug logging"),
    overwrite: bool = typer.Option(False, help="Overwrite existing processed files")
):
    """Process whole-slide images and extract patches."""
    
    # Setup logging
    setup_logging(level="DEBUG" if debug else "INFO") 
    logger = logging.getLogger(__name__)
    
    # Parse parameters
    magnifications_list = [float(x.strip()) for x in magnifications.split(",")]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting slide preprocessing")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Magnifications: {magnifications_list}")
    logger.info(f"Patch size: {patch_size}")
    logger.info(f"Workers: {num_workers}")
    
    # Find slide files
    slide_files = []
    for ext in ["*.svs", "*.tiff", "*.ndpi", "*.mrxs"]:
        slide_files.extend(input_path.glob(ext))
        
    logger.info(f"Found {len(slide_files)} slide files")
    
    if len(slide_files) == 0:
        logger.warning("No slide files found!")
        return
        
    # Filter out already processed files if not overwriting
    if not overwrite:
        remaining_files = []
        for slide_file in slide_files:
            slide_id = slide_file.stem
            output_file = output_path / f"{slide_id}_processed.{output_format}"
            if not output_file.exists():
                remaining_files.append(slide_file)
            else:
                logger.info(f"Skipping {slide_id} (already processed)")
        slide_files = remaining_files
        
    logger.info(f"Processing {len(slide_files)} slides")
    
    # Setup slide processor
    slide_processor = SlideProcessor(
        patch_size=patch_size,
        overlap=overlap,
        tissue_threshold=tissue_threshold,
        background_threshold=background_threshold,
        min_tissue_area=min_tissue_area,
        normalize_stains=normalize_stains,
        save_patches=save_patches,
        output_dir=str(output_path / "patches") if save_patches else None
    )
    
    # Process slides
    if num_workers == 1:
        # Serial processing
        for slide_file in tqdm(slide_files, desc="Processing slides"):
            _process_single_slide(
                slide_file, slide_processor, magnifications_list, 
                max_patches, output_path, output_format, save_thumbnails, logger
            )
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for slide_file in slide_files:
                future = executor.submit(
                    _process_single_slide,
                    slide_file, slide_processor, magnifications_list,
                    max_patches, output_path, output_format, save_thumbnails, logger
                )
                futures.append(future)
                
            # Process completed futures
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slides"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Failed to process slide: {e}")
                    
    logger.info("Slide preprocessing completed!")


def _process_single_slide(
    slide_file: Path,
    slide_processor: SlideProcessor,
    magnifications: List[float],
    max_patches: Optional[int],
    output_path: Path,
    output_format: str,
    save_thumbnails: bool,
    logger
):
    """Process a single slide."""
    
    slide_id = slide_file.stem
    
    try:
        # Process slide
        slide_data = slide_processor.process_slide(
            slide_file, magnifications, max_patches
        )
        
        # Save processed data
        if output_format == "h5":
            output_file = output_path / f"{slide_id}_processed.h5"
            slide_processor.save_slide_data(slide_data, output_file)
        elif output_format == "pt":
            output_file = output_path / f"{slide_id}_processed.pt"
            torch.save(slide_data, output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        # Save thumbnail if requested
        if save_thumbnails and slide_data.thumbnail is not None:
            import cv2
            thumbnail_file = output_path / "thumbnails" / f"{slide_id}_thumbnail.jpg"
            thumbnail_file.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(thumbnail_file), cv2.cvtColor(slide_data.thumbnail, cv2.COLOR_RGB2BGR))
            
        logger.info(f"Processed {slide_id}: {len(slide_data.patches)} patches")
        
    except Exception as e:
        logger.error(f"Failed to process {slide_id}: {e}")
        raise


@app.command()
def build_graphs(
    input_dir: str = typer.Option(..., help="Directory containing processed slide data"),
    output_dir: str = typer.Option(..., help="Output directory for graphs"),
    
    # Graph building parameters
    feature_extractor: str = typer.Option("dinov2", help="Feature extraction method"),
    spatial_k: int = typer.Option(8, help="Number of spatial neighbors"),
    morphological_k: int = typer.Option(16, help="Number of morphological neighbors"),
    edge_threshold: float = typer.Option(0.7, help="Edge creation threshold"),
    
    # Processing options
    num_workers: int = typer.Option(4, help="Number of parallel workers"),
    batch_size: int = typer.Option(1, help="Batch size for feature extraction"),
    
    # Output options
    save_hierarchical: bool = typer.Option(False, help="Save hierarchical graphs"),
    
    # Misc
    debug: bool = typer.Option(False, help="Enable debug logging"),
    overwrite: bool = typer.Option(False, help="Overwrite existing graph files")
):
    """Build tissue graphs from processed slide data."""
    
    # Setup logging
    setup_logging(level="DEBUG" if debug else "INFO")
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting graph building")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Feature extractor: {feature_extractor}")
    
    # Find processed slide files
    processed_files = []
    for ext in ["*.h5", "*.pt"]:
        processed_files.extend(input_path.glob(ext))
        
    logger.info(f"Found {len(processed_files)} processed slide files")
    
    # Filter out already processed files if not overwriting
    if not overwrite:
        remaining_files = []
        for processed_file in processed_files:
            slide_id = processed_file.stem.replace("_processed", "")
            graph_file = output_path / f"{slide_id}_graph.pt"
            if not graph_file.exists():
                remaining_files.append(processed_file)
            else:
                logger.info(f"Skipping {slide_id} (graph already exists)")
        processed_files = remaining_files
        
    logger.info(f"Building graphs for {len(processed_files)} slides")
    
    # Setup graph builder
    graph_builder = TissueGraphBuilder(
        feature_extractor=feature_extractor,
        spatial_k=spatial_k,
        morphological_k=morphological_k,
        edge_threshold=edge_threshold
    )
    
    # Build graphs
    for processed_file in tqdm(processed_files, desc="Building graphs"):
        try:
            _build_single_graph(
                processed_file, graph_builder, output_path, 
                save_hierarchical, logger
            )
        except Exception as e:
            logger.error(f"Failed to build graph for {processed_file}: {e}")
            
    logger.info("Graph building completed!")


def _build_single_graph(
    processed_file: Path,
    graph_builder: TissueGraphBuilder,
    output_path: Path,
    save_hierarchical: bool,
    logger
):
    """Build graph for a single slide."""
    
    slide_id = processed_file.stem.replace("_processed", "")
    
    # Load processed slide data
    if processed_file.suffix == ".h5":
        from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
        slide_data = SlideProcessor.load_slide_data(processed_file)
    elif processed_file.suffix == ".pt":
        slide_data = torch.load(processed_file)
    else:
        raise ValueError(f"Unsupported file format: {processed_file.suffix}")
        
    # Build graph
    graph = graph_builder.build_graph(slide_data)
    
    # Save graph
    graph_file = output_path / f"{slide_id}_graph.pt"
    torch.save(graph, graph_file)
    
    # Save hierarchical graphs if requested
    if save_hierarchical:
        hierarchical_graphs = graph_builder.create_hierarchical_graph(graph, levels=3)
        hier_file = output_path / f"{slide_id}_hierarchical.pt"
        torch.save(hierarchical_graphs, hier_file)
        
    logger.info(f"Built graph for {slide_id}: {graph.num_nodes} nodes, {graph.num_edges} edges")


@app.command()
def validate_preprocessing(
    data_dir: str = typer.Option(..., help="Directory containing processed data"),
    output_file: str = typer.Option("validation_report.txt", help="Output validation report")
):
    """Validate preprocessed data quality."""
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    data_path = Path(data_dir)
    
    # Find all processed files
    processed_files = list(data_path.glob("*_processed.*"))
    graph_files = list(data_path.glob("*_graph.pt"))
    
    logger.info(f"Validating {len(processed_files)} processed files")
    logger.info(f"Validating {len(graph_files)} graph files")
    
    # Validation results
    validation_results = {
        "total_processed": len(processed_files),
        "total_graphs": len(graph_files),
        "valid_processed": 0,
        "valid_graphs": 0,
        "errors": []
    }
    
    # Validate processed files
    for processed_file in tqdm(processed_files, desc="Validating processed files"):
        try:
            if processed_file.suffix == ".h5":
                from dgdm_histopath.preprocessing.slide_processor import SlideProcessor
                slide_data = SlideProcessor.load_slide_data(processed_file)
            elif processed_file.suffix == ".pt":
                slide_data = torch.load(processed_file)
            else:
                continue
                
            # Basic validation
            if len(slide_data.patches) > 0:
                validation_results["valid_processed"] += 1
            else:
                validation_results["errors"].append(f"No patches in {processed_file}")
                
        except Exception as e:
            validation_results["errors"].append(f"Error loading {processed_file}: {e}")
            
    # Validate graph files
    for graph_file in tqdm(graph_files, desc="Validating graph files"):
        try:
            graph = torch.load(graph_file)
            
            # Basic validation
            if graph.num_nodes > 0 and graph.x.size(0) == graph.num_nodes:
                validation_results["valid_graphs"] += 1
            else:
                validation_results["errors"].append(f"Invalid graph structure in {graph_file}")
                
        except Exception as e:
            validation_results["errors"].append(f"Error loading {graph_file}: {e}")
            
    # Save validation report
    with open(output_file, "w") as f:
        f.write("DGDM Preprocessing Validation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total processed files: {validation_results['total_processed']}\n")
        f.write(f"Valid processed files: {validation_results['valid_processed']}\n")
        f.write(f"Total graph files: {validation_results['total_graphs']}\n")
        f.write(f"Valid graph files: {validation_results['valid_graphs']}\n")
        f.write(f"Total errors: {len(validation_results['errors'])}\n\n")
        
        if validation_results['errors']:
            f.write("Errors:\n")
            for error in validation_results['errors']:
                f.write(f"- {error}\n")
                
    logger.info(f"Validation completed. Report saved to {output_file}")


if __name__ == "__main__":
    app()