"""
PyTorch Lightning data module for histopathology datasets.

Handles data loading, preprocessing, and batching for DGDM training.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from typing import Optional, Dict, List, Union
import os
from pathlib import Path
import logging

from dgdm_histopath.data.dataset import HistopathDataset, SlideDataset


class HistopathDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for histopathology data.
    
    Handles automatic data loading, splitting, and batching for 
    graph-based histopathology analysis with DGDM.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        dataset_type: str = "slide",
        batch_size: int = 4,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        augmentations: str = "none",
        max_slides_per_split: Optional[int] = None,
        cache_graphs: bool = True,
        shuffle_train: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize histopathology data module.
        
        Args:
            data_dir: Directory containing histopathology data
            dataset_type: Type of dataset ("slide", "patch", "graph")
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
            prefetch_factor: Number of samples loaded in advance by each worker
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            augmentations: Augmentation strategy ("none", "light", "strong")
            max_slides_per_split: Maximum slides per data split
            cache_graphs: Whether to cache processed graphs
            shuffle_train: Whether to shuffle training data
            drop_last: Whether to drop last incomplete batch
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.augmentations = augmentations
        self.max_slides_per_split = max_slides_per_split
        self.cache_graphs = cache_graphs
        self.shuffle_train = shuffle_train
        self.drop_last = drop_last
        
        # Validate splits
        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Data splits must sum to 1.0")
            
        self.logger = logging.getLogger(__name__)
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """
        Download and prepare data (called only on main process).
        This method should not assign state (don't set self.x = y).
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
            
        self.logger.info(f"Data directory exists: {self.data_dir}")
        
        # Count available slides/samples
        if self.dataset_type == "slide":
            slide_files = list(self.data_dir.glob("*.svs")) + list(self.data_dir.glob("*.tiff"))
            self.logger.info(f"Found {len(slide_files)} slide files")
        elif self.dataset_type == "graph":
            graph_files = list(self.data_dir.glob("*.h5")) + list(self.data_dir.glob("*.pt"))
            self.logger.info(f"Found {len(graph_files)} graph files")
            
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for each stage (fit/validate/test/predict).
        
        Args:
            stage: Current stage ("fit", "validate", "test", "predict", None)
        """
        if stage == "fit" or stage is None:
            # Create full dataset
            if self.dataset_type == "slide":
                full_dataset = SlideDataset(
                    data_dir=self.data_dir,
                    augmentations=self.augmentations,
                    cache_graphs=self.cache_graphs
                )
            else:
                full_dataset = HistopathDataset(
                    data_dir=self.data_dir,
                    dataset_type=self.dataset_type,
                    augmentations=self.augmentations,
                    cache_graphs=self.cache_graphs
                )
                
            # Split dataset
            total_size = len(full_dataset)
            train_size = int(self.train_split * total_size)
            val_size = int(self.val_split * total_size)
            test_size = total_size - train_size - val_size
            
            # Apply maximum slide limits if specified
            if self.max_slides_per_split is not None:
                train_size = min(train_size, self.max_slides_per_split)
                val_size = min(val_size, self.max_slides_per_split)
                test_size = min(test_size, self.max_slides_per_split)
                
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
            
            self.logger.info(f"Dataset splits - Train: {len(self.train_dataset)}, "
                           f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
            
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                # Create test dataset if not already created
                if self.dataset_type == "slide":
                    test_dataset = SlideDataset(
                        data_dir=self.data_dir,
                        augmentations="none",  # No augmentations for test
                        cache_graphs=self.cache_graphs
                    )
                else:
                    test_dataset = HistopathDataset(
                        data_dir=self.data_dir,
                        dataset_type=self.dataset_type,
                        augmentations="none",
                        cache_graphs=self.cache_graphs
                    )
                    
                # Use full dataset as test if splits not created
                self.test_dataset = test_dataset
                
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if self.dataset_type in ["graph", "slide"]:
            return GeometricDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
                drop_last=self.drop_last
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_train,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2,
                drop_last=self.drop_last
            )
            
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if self.dataset_type in ["graph", "slide"]:
            return GeometricDataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2
            )
            
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if self.dataset_type in ["graph", "slide"]:
            return GeometricDataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else 2
            )
            
    def predict_dataloader(self) -> DataLoader:
        """Create prediction data loader."""
        return self.test_dataloader()
        
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset."""
        info = {
            "data_dir": str(self.data_dir),
            "dataset_type": self.dataset_type,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split
            }
        }
        
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            info["dataset_sizes"] = {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset) if self.val_dataset else 0,
                "test": len(self.test_dataset) if self.test_dataset else 0
            }
            
        return info
        
    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        # Clean up any cached data or temporary files
        self.logger.info(f"Tearing down data module for stage: {stage}")
        
    @staticmethod
    def add_data_args(parser):
        """Add data-related arguments to argument parser."""
        group = parser.add_argument_group("Data")
        
        group.add_argument("--data_dir", type=str, required=True,
                          help="Directory containing histopathology data")
        group.add_argument("--dataset_type", type=str, default="slide",
                          choices=["slide", "patch", "graph"],
                          help="Type of dataset")
        group.add_argument("--batch_size", type=int, default=4,
                          help="Batch size for training")
        group.add_argument("--num_workers", type=int, default=8,
                          help="Number of data loading workers")
        group.add_argument("--train_split", type=float, default=0.7,
                          help="Fraction of data for training")
        group.add_argument("--val_split", type=float, default=0.15,
                          help="Fraction of data for validation")
        group.add_argument("--test_split", type=float, default=0.15,
                          help="Fraction of data for testing")
        group.add_argument("--augmentations", type=str, default="none",
                          choices=["none", "light", "strong"],
                          help="Augmentation strategy")
        group.add_argument("--max_slides_per_split", type=int, default=None,
                          help="Maximum slides per data split")
        group.add_argument("--cache_graphs", action="store_true",
                          help="Cache processed graphs")
        group.add_argument("--no_shuffle_train", action="store_true",
                          help="Don't shuffle training data")
        
        return group