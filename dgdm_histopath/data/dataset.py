"""
PyTorch datasets for histopathology data.

Implements datasets for loading and processing whole-slide images,
patches, and graph representations.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from pathlib import Path
import h5py
import pickle
import logging
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from dgdm_histopath.preprocessing.slide_processor import SlideProcessor, SlideData
from dgdm_histopath.preprocessing.tissue_graph_builder import TissueGraphBuilder


class HistopathDataset(Dataset):
    """
    Base dataset class for histopathology data.
    
    Handles loading of preprocessed graphs, patches, or slide-level data
    with optional augmentations and caching.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_type: str = "graph",
        augmentations: str = "none",
        cache_graphs: bool = True,
        max_samples: Optional[int] = None,
        metadata_file: Optional[str] = None
    ):
        """
        Initialize histopathology dataset.
        
        Args:
            data_dir: Directory containing data files
            dataset_type: Type of data ("graph", "patch", "slide")
            augmentations: Augmentation strategy
            cache_graphs: Whether to cache loaded graphs in memory
            max_samples: Maximum number of samples to load
            metadata_file: Optional metadata file with labels/targets
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.augmentations = augmentations
        self.cache_graphs = cache_graphs
        self.max_samples = max_samples
        
        self.logger = logging.getLogger(__name__)
        
        # Find data files
        self.data_files = self._find_data_files()
        
        if max_samples is not None:
            self.data_files = self.data_files[:max_samples]
            
        self.logger.info(f"Found {len(self.data_files)} {dataset_type} files")
        
        # Load metadata if provided
        self.metadata = {}
        if metadata_file:
            metadata_path = self.data_dir / metadata_file
            if metadata_path.exists():
                self.metadata = self._load_metadata(metadata_path)
                
        # Cache for loaded graphs
        self.graph_cache = {} if cache_graphs else None
        
        # Setup augmentations
        self.transform = self._create_transforms()
        
    def _find_data_files(self) -> List[Path]:
        """Find data files in the directory."""
        if self.dataset_type == "graph":
            # Look for graph files (.h5, .pt, .pkl)
            files = []
            files.extend(self.data_dir.glob("*.h5"))
            files.extend(self.data_dir.glob("*.pt"))
            files.extend(self.data_dir.glob("*.pkl"))
        elif self.dataset_type == "patch":
            # Look for image files
            files = []
            files.extend(self.data_dir.glob("*.png"))
            files.extend(self.data_dir.glob("*.jpg"))
            files.extend(self.data_dir.glob("*.jpeg"))
        elif self.dataset_type == "slide":
            # Look for slide files
            files = []
            files.extend(self.data_dir.glob("*.svs"))
            files.extend(self.data_dir.glob("*.tiff"))
            files.extend(self.data_dir.glob("*.ndpi"))
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        return sorted(files)
        
    def _load_metadata(self, metadata_path: Path) -> Dict:
        """Load metadata from file."""
        if metadata_path.suffix == ".json":
            with open(metadata_path, 'r') as f:
                return json.load(f)
        elif metadata_path.suffix == ".csv":
            import pandas as pd
            df = pd.read_csv(metadata_path)
            return df.to_dict('records')
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")
            
    def _create_transforms(self) -> Optional[A.Compose]:
        """Create augmentation transforms."""
        if self.augmentations == "none":
            return None
        elif self.augmentations == "light":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.3),
                ToTensorV2()
            ])
        elif self.augmentations == "strong":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                ToTensorV2()
            ])
        else:
            return None
            
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_files)
        
    def __getitem__(self, idx: int) -> Union[Data, Dict[str, Any]]:
        """
        Get item from dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Data sample (graph, patch, or slide data)
        """
        file_path = self.data_files[idx]
        sample_id = file_path.stem
        
        # Check cache first
        if self.graph_cache is not None and sample_id in self.graph_cache:
            return self.graph_cache[sample_id]
            
        # Load data based on type
        if self.dataset_type == "graph":
            sample = self._load_graph(file_path)
        elif self.dataset_type == "patch":
            sample = self._load_patch(file_path)
        elif self.dataset_type == "slide":
            sample = self._load_slide(file_path)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        # Add metadata if available
        if sample_id in self.metadata:
            sample.update(self.metadata[sample_id])
            
        # Cache if enabled
        if self.graph_cache is not None:
            self.graph_cache[sample_id] = sample
            
        return sample
        
    def _load_graph(self, file_path: Path) -> Data:
        """Load graph data from file."""
        if file_path.suffix == ".h5":
            return self._load_graph_h5(file_path)
        elif file_path.suffix == ".pt":
            return torch.load(file_path)
        elif file_path.suffix == ".pkl":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported graph format: {file_path.suffix}")
            
    def _load_graph_h5(self, file_path: Path) -> Data:
        """Load graph from HDF5 file."""
        with h5py.File(file_path, 'r') as f:
            # Load node features
            node_features = torch.tensor(f['node_features'][:], dtype=torch.float)
            
            # Load edge indices
            edge_index = torch.tensor(f['edge_index'][:], dtype=torch.long)
            
            # Load edge attributes if available
            edge_attr = None
            if 'edge_attr' in f:
                edge_attr = torch.tensor(f['edge_attr'][:], dtype=torch.float)
                
            # Load node positions if available
            pos = None
            if 'node_pos' in f:
                pos = torch.tensor(f['node_pos'][:], dtype=torch.float)
                
            # Load labels if available
            y = None
            if 'labels' in f:
                y = torch.tensor(f['labels'][:], dtype=torch.long)
                
            # Create Data object
            data = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                y=y
            )
            
            # Add metadata
            if 'metadata' in f:
                for key in f['metadata'].attrs:
                    setattr(data, key, f['metadata'].attrs[key])
                    
        return data
        
    def _load_patch(self, file_path: Path) -> Dict[str, Any]:
        """Load patch image data."""
        # Load image
        image = Image.open(file_path).convert('RGB')
        image_array = np.array(image)
        
        # Apply transforms if available
        if self.transform is not None:
            transformed = self.transform(image=image_array)
            image_tensor = transformed['image']
        else:
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            
        return {
            'image': image_tensor,
            'file_path': str(file_path),
            'patch_id': file_path.stem
        }
        
    def _load_slide(self, file_path: Path) -> Dict[str, Any]:
        """Load slide-level data (placeholder - requires full slide processing)."""
        # This is a placeholder - full slide loading would require
        # integration with SlideProcessor and TissueGraphBuilder
        return {
            'slide_path': str(file_path),
            'slide_id': file_path.stem,
            'processed': False  # Indicates slide needs processing
        }


class SlideDataset(HistopathDataset):
    """
    Dataset for whole-slide images with on-the-fly graph construction.
    
    Processes slides on demand and builds tissue graphs for training.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        patch_size: int = 256,
        magnifications: List[float] = [20.0],
        tissue_threshold: float = 0.8,
        max_patches: Optional[int] = 1000,
        feature_extractor: str = "dinov2",
        augmentations: str = "none",
        cache_graphs: bool = True,
        preprocess_slides: bool = False,
        preprocessed_dir: Optional[str] = None
    ):
        """
        Initialize slide dataset.
        
        Args:
            data_dir: Directory containing slide files
            patch_size: Size of patches to extract
            magnifications: Target magnifications for analysis
            tissue_threshold: Minimum tissue percentage for patches
            max_patches: Maximum patches per slide
            feature_extractor: Feature extraction method
            augmentations: Augmentation strategy
            cache_graphs: Whether to cache processed graphs
            preprocess_slides: Whether to preprocess slides
            preprocessed_dir: Directory for preprocessed data
        """
        # Initialize base dataset
        super().__init__(
            data_dir=data_dir,
            dataset_type="slide",
            augmentations=augmentations,
            cache_graphs=cache_graphs
        )
        
        self.patch_size = patch_size
        self.magnifications = magnifications
        self.tissue_threshold = tissue_threshold
        self.max_patches = max_patches
        self.feature_extractor = feature_extractor
        self.preprocess_slides = preprocess_slides
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        
        # Initialize slide processor and graph builder
        self.slide_processor = SlideProcessor(
            patch_size=patch_size,
            tissue_threshold=tissue_threshold,
            save_patches=False
        )
        
        self.graph_builder = TissueGraphBuilder(
            feature_extractor=feature_extractor
        )
        
        # Preprocess slides if requested
        if preprocess_slides:
            self._preprocess_all_slides()
            
    def _preprocess_all_slides(self):
        """Preprocess all slides and save graphs."""
        if self.preprocessed_dir is None:
            self.preprocessed_dir = self.data_dir / "preprocessed"
            
        self.preprocessed_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Preprocessing {len(self.data_files)} slides...")
        
        for i, slide_path in enumerate(self.data_files):
            slide_id = slide_path.stem
            output_path = self.preprocessed_dir / f"{slide_id}_graph.pt"
            
            if output_path.exists():
                self.logger.info(f"Skipping {slide_id} (already processed)")
                continue
                
            try:
                # Process slide
                slide_data = self.slide_processor.process_slide(
                    slide_path, self.magnifications, self.max_patches
                )
                
                # Build graph
                graph = self.graph_builder.build_graph(slide_data)
                
                # Save graph
                torch.save(graph, output_path)
                
                self.logger.info(f"Processed slide {i+1}/{len(self.data_files)}: {slide_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to process slide {slide_id}: {e}")
                
    def __getitem__(self, idx: int) -> Data:
        """
        Get processed graph for slide.
        
        Args:
            idx: Sample index
            
        Returns:
            Tissue graph as PyTorch Geometric Data object
        """
        file_path = self.data_files[idx]
        slide_id = file_path.stem
        
        # Check cache first
        if self.graph_cache is not None and slide_id in self.graph_cache:
            return self.graph_cache[slide_id]
            
        # Check for preprocessed graph
        if self.preprocessed_dir is not None:
            graph_path = self.preprocessed_dir / f"{slide_id}_graph.pt"
            if graph_path.exists():
                graph = torch.load(graph_path)
                
                # Cache if enabled
                if self.graph_cache is not None:
                    self.graph_cache[slide_id] = graph
                    
                return graph
                
        # Process slide on-the-fly
        try:
            # Process slide
            slide_data = self.slide_processor.process_slide(
                file_path, self.magnifications, self.max_patches
            )
            
            # Build graph
            graph = self.graph_builder.build_graph(slide_data)
            
            # Add slide metadata
            graph.slide_id = slide_id
            graph.slide_path = str(file_path)
            
            # Cache if enabled
            if self.graph_cache is not None:
                self.graph_cache[slide_id] = graph
                
            return graph
            
        except Exception as e:
            self.logger.error(f"Failed to process slide {slide_id}: {e}")
            # Return empty graph as fallback
            return Data(
                x=torch.empty((0, 768), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=0,
                slide_id=slide_id,
                error=str(e)
            )


class GraphDataset(HistopathDataset):
    """
    Dataset for preprocessed tissue graphs.
    
    Loads precomputed graph representations with optional
    augmentations and multi-level hierarchical graphs.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        hierarchical: bool = False,
        max_nodes: Optional[int] = None,
        node_feature_dim: int = 768,
        **kwargs
    ):
        """
        Initialize graph dataset.
        
        Args:
            data_dir: Directory containing graph files
            hierarchical: Whether to load hierarchical graphs
            max_nodes: Maximum nodes per graph (for memory efficiency)
            node_feature_dim: Expected node feature dimension
            **kwargs: Additional arguments for base class
        """
        super().__init__(data_dir=data_dir, dataset_type="graph", **kwargs)
        
        self.hierarchical = hierarchical
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        
    def __getitem__(self, idx: int) -> Union[Data, List[Data]]:
        """
        Get graph sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Graph data (single graph or list for hierarchical)
        """
        graph = super().__getitem__(idx)
        
        # Subsample nodes if graph is too large
        if self.max_nodes is not None and graph.num_nodes > self.max_nodes:
            graph = self._subsample_graph(graph)
            
        # Load hierarchical graphs if requested
        if self.hierarchical:
            return self._load_hierarchical_graphs(graph)
        else:
            return graph
            
    def _subsample_graph(self, graph: Data) -> Data:
        """Subsample graph to maximum number of nodes."""
        num_nodes = graph.num_nodes
        
        # Random sampling of nodes
        perm = torch.randperm(num_nodes)[:self.max_nodes]
        
        # Update node features and positions
        new_x = graph.x[perm]
        new_pos = graph.pos[perm] if hasattr(graph, 'pos') and graph.pos is not None else None
        
        # Update edge index
        node_map = torch.full((num_nodes,), -1, dtype=torch.long)
        node_map[perm] = torch.arange(self.max_nodes)
        
        edge_mask = (node_map[graph.edge_index[0]] >= 0) & (node_map[graph.edge_index[1]] >= 0)
        new_edge_index = graph.edge_index[:, edge_mask]
        new_edge_index = node_map[new_edge_index]
        
        # Update edge attributes
        new_edge_attr = graph.edge_attr[edge_mask] if hasattr(graph, 'edge_attr') and graph.edge_attr is not None else None
        
        # Create new graph
        new_graph = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            pos=new_pos,
            num_nodes=self.max_nodes
        )
        
        # Copy other attributes
        for key, value in graph:
            if key not in ['x', 'edge_index', 'edge_attr', 'pos', 'num_nodes']:
                setattr(new_graph, key, value)
                
        return new_graph
        
    def _load_hierarchical_graphs(self, base_graph: Data) -> List[Data]:
        """Create hierarchical representation from base graph."""
        # This would implement hierarchical graph construction
        # For now, return single graph
        return [base_graph]