"""
Whole-slide image preprocessing for histopathology analysis.

Handles loading, tiling, and preprocessing of gigapixel WSI files
with support for multiple magnification levels and tissue detection.
"""

import os
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
import h5py
from dataclasses import dataclass
import logging
from pathlib import Path

# OpenSlide imports (with fallback for systems without openslide)
try:
    import openslide
    from openslide import OpenSlide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    OpenSlide = None  # Define as None for type hints
    logging.warning("OpenSlide not available. Some WSI formats may not be supported.")

from dgdm_histopath.preprocessing.tissue_detection import TissueDetector
from dgdm_histopath.preprocessing.stain_normalization import StainNormalizer


@dataclass
class PatchInfo:
    """Information about a tissue patch."""
    x: int
    y: int
    level: int
    magnification: float
    patch_id: str
    tissue_percentage: float
    features: Optional[np.ndarray] = None


@dataclass
class SlideData:
    """Container for processed slide data."""
    slide_id: str
    patches: List[PatchInfo]
    metadata: Dict
    thumbnail: Optional[np.ndarray] = None
    tissue_mask: Optional[np.ndarray] = None


class SlideProcessor:
    """
    High-performance whole-slide image processor for histopathology.
    
    Supports multiple formats (SVS, TIFF, NDPI, etc.) and handles
    tissue detection, patch extraction, and preprocessing pipelines.
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        overlap: float = 0.0,
        tissue_threshold: float = 0.8,
        background_threshold: int = 220,
        min_tissue_area: int = 1000,
        normalize_stains: bool = True,
        save_patches: bool = False,
        output_dir: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize slide processor.
        
        Args:
            patch_size: Size of extracted patches in pixels
            overlap: Overlap between adjacent patches (0.0 to 1.0)
            tissue_threshold: Minimum tissue percentage to keep patch
            background_threshold: Background intensity threshold (0-255)
            min_tissue_area: Minimum tissue area in pixels
            normalize_stains: Whether to apply stain normalization
            save_patches: Whether to save extracted patches to disk
            output_dir: Directory to save patches (if save_patches=True)
            use_gpu: Whether to use GPU acceleration when available
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        self.background_threshold = background_threshold
        self.min_tissue_area = min_tissue_area
        self.normalize_stains = normalize_stains
        self.save_patches = save_patches
        self.output_dir = Path(output_dir) if output_dir else None
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize components
        self.tissue_detector = TissueDetector(
            background_threshold=background_threshold,
            min_area=min_tissue_area
        )
        
        if normalize_stains:
            self.stain_normalizer = StainNormalizer()
        else:
            self.stain_normalizer = None
            
        # Setup output directory
        if self.save_patches and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self.logger = logging.getLogger(__name__)
        
    def load_slide(self, slide_path: Union[str, Path]) -> OpenSlide:
        """
        Load a whole-slide image.
        
        Args:
            slide_path: Path to the slide file
            
        Returns:
            OpenSlide object
            
        Raises:
            RuntimeError: If OpenSlide is not available or slide cannot be loaded
        """
        if not OPENSLIDE_AVAILABLE:
            raise RuntimeError(
                "OpenSlide is required for WSI processing. "
                "Install with: pip install openslide-python"
            )
            
        slide_path = Path(slide_path)
        if not slide_path.exists():
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
            
        try:
            slide = OpenSlide(str(slide_path))
            self.logger.info(f"Loaded slide: {slide_path}")
            self.logger.info(f"Dimensions: {slide.dimensions}")
            self.logger.info(f"Levels: {slide.level_count}")
            return slide
        except Exception as e:
            raise RuntimeError(f"Failed to load slide {slide_path}: {e}")
            
    def get_slide_metadata(self, slide: OpenSlide) -> Dict:
        """Extract metadata from slide."""
        metadata = {
            'dimensions': slide.dimensions,
            'level_count': slide.level_count,
            'level_dimensions': [slide.level_dimensions[i] for i in range(slide.level_count)],
            'level_downsamples': [slide.level_downsamples[i] for i in range(slide.level_count)],
            'properties': dict(slide.properties),
        }
        
        # Extract magnification if available
        if 'openslide.objective-power' in slide.properties:
            metadata['objective_power'] = float(slide.properties['openslide.objective-power'])
        elif 'aperio.AppMag' in slide.properties:
            metadata['objective_power'] = float(slide.properties['aperio.AppMag'])
        else:
            metadata['objective_power'] = 40.0  # Default assumption
            
        return metadata
        
    def get_thumbnail(self, slide: OpenSlide, max_size: int = 1024) -> np.ndarray:
        """
        Get slide thumbnail.
        
        Args:
            slide: OpenSlide object
            max_size: Maximum thumbnail dimension
            
        Returns:
            Thumbnail as RGB numpy array
        """
        # Calculate thumbnail size maintaining aspect ratio
        width, height = slide.dimensions
        aspect_ratio = width / height
        
        if aspect_ratio > 1:
            thumb_width = max_size
            thumb_height = int(max_size / aspect_ratio)
        else:
            thumb_width = int(max_size * aspect_ratio)
            thumb_height = max_size
            
        # Get thumbnail
        thumbnail = slide.get_thumbnail((thumb_width, thumb_height))
        return np.array(thumbnail)
        
    def detect_tissue_regions(
        self, 
        slide: OpenSlide, 
        level: int = -1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect tissue regions in the slide.
        
        Args:
            slide: OpenSlide object
            level: Pyramid level to use for detection (-1 for lowest resolution)
            
        Returns:
            Tuple of (tissue_mask, tissue_image)
        """
        if level == -1:
            level = slide.level_count - 1
            
        # Get low-resolution image for tissue detection
        level_dims = slide.level_dimensions[level]
        tissue_image = slide.read_region((0, 0), level, level_dims)
        tissue_image = np.array(tissue_image.convert('RGB'))
        
        # Detect tissue
        tissue_mask = self.tissue_detector.detect_tissue(tissue_image)
        
        return tissue_mask, tissue_image
        
    def generate_patch_coordinates(
        self,
        slide: OpenSlide,
        magnifications: List[float],
        tissue_mask: np.ndarray,
        mask_level: int = -1
    ) -> List[Tuple[int, int, int, float]]:
        """
        Generate patch coordinates for tissue regions.
        
        Args:
            slide: OpenSlide object
            magnifications: List of target magnifications
            tissue_mask: Binary tissue mask
            mask_level: Level used for tissue mask
            
        Returns:
            List of (x, y, level, magnification) tuples
        """
        if mask_level == -1:
            mask_level = slide.level_count - 1
            
        coordinates = []
        base_magnification = self.get_slide_metadata(slide)['objective_power']
        
        for target_mag in magnifications:
            # Find best level for target magnification
            level = self._find_best_level(slide, target_mag, base_magnification)
            level_downsample = slide.level_downsamples[level]
            actual_mag = base_magnification / level_downsample
            
            # Calculate step size considering overlap
            step_size = int(self.patch_size * (1 - self.overlap))
            
            # Scale coordinates from mask level to target level
            mask_downsample = slide.level_downsamples[mask_level]
            scale_factor = mask_downsample / level_downsample
            
            # Generate coordinates
            mask_height, mask_width = tissue_mask.shape
            for mask_y in range(0, mask_height, step_size):
                for mask_x in range(0, mask_width, step_size):
                    # Check if region contains tissue
                    patch_mask = tissue_mask[
                        mask_y:mask_y + step_size,
                        mask_x:mask_x + step_size
                    ]
                    
                    if patch_mask.size == 0:
                        continue
                        
                    tissue_percentage = patch_mask.mean()
                    if tissue_percentage >= self.tissue_threshold:
                        # Convert to level 0 coordinates
                        x = int(mask_x * mask_downsample)
                        y = int(mask_y * mask_downsample)
                        
                        coordinates.append((x, y, level, actual_mag))
                        
        self.logger.info(f"Generated {len(coordinates)} patch coordinates")
        return coordinates
        
    def _find_best_level(
        self, 
        slide: OpenSlide, 
        target_magnification: float, 
        base_magnification: float
    ) -> int:
        """Find the best pyramid level for target magnification."""
        target_downsample = base_magnification / target_magnification
        
        best_level = 0
        min_diff = float('inf')
        
        for level in range(slide.level_count):
            level_downsample = slide.level_downsamples[level]
            diff = abs(level_downsample - target_downsample)
            
            if diff < min_diff:
                min_diff = diff
                best_level = level
                
        return best_level
        
    def extract_patch(
        self, 
        slide: OpenSlide, 
        x: int, 
        y: int, 
        level: int
    ) -> Optional[np.ndarray]:
        """
        Extract a single patch from the slide.
        
        Args:
            slide: OpenSlide object
            x, y: Coordinates in level 0
            level: Pyramid level
            
        Returns:
            Patch as RGB numpy array or None if extraction fails
        """
        try:
            # Read patch from slide
            patch = slide.read_region((x, y), level, (self.patch_size, self.patch_size))
            patch = patch.convert('RGB')
            patch_array = np.array(patch)
            
            # Apply stain normalization if enabled
            if self.stain_normalizer is not None:
                patch_array = self.stain_normalizer.normalize(patch_array)
                
            return patch_array
            
        except Exception as e:
            self.logger.warning(f"Failed to extract patch at ({x}, {y}): {e}")
            return None
            
    def process_slide(
        self,
        slide_path: Union[str, Path],
        magnifications: List[float] = [5.0, 20.0, 40.0],
        max_patches: Optional[int] = None
    ) -> SlideData:
        """
        Process a complete slide.
        
        Args:
            slide_path: Path to slide file
            magnifications: List of target magnifications
            max_patches: Maximum number of patches to extract
            
        Returns:
            SlideData object containing processed information
        """
        slide_path = Path(slide_path)
        slide_id = slide_path.stem
        
        self.logger.info(f"Processing slide: {slide_id}")
        
        # Load slide
        slide = self.load_slide(slide_path)
        
        try:
            # Get metadata and thumbnail
            metadata = self.get_slide_metadata(slide)
            thumbnail = self.get_thumbnail(slide)
            
            # Detect tissue regions
            tissue_mask, tissue_image = self.detect_tissue_regions(slide)
            
            # Generate patch coordinates
            coordinates = self.generate_patch_coordinates(
                slide, magnifications, tissue_mask
            )
            
            # Limit number of patches if specified
            if max_patches and len(coordinates) > max_patches:
                # Sample patches uniformly
                indices = np.linspace(0, len(coordinates) - 1, max_patches, dtype=int)
                coordinates = [coordinates[i] for i in indices]
                
            # Extract patches
            patches = []
            for i, (x, y, level, magnification) in enumerate(coordinates):
                patch_array = self.extract_patch(slide, x, y, level)
                
                if patch_array is not None:
                    # Calculate tissue percentage for this patch
                    patch_tissue_pct = self._calculate_tissue_percentage(patch_array)
                    
                    patch_info = PatchInfo(
                        x=x,
                        y=y,
                        level=level,
                        magnification=magnification,
                        patch_id=f"{slide_id}_patch_{i:06d}",
                        tissue_percentage=patch_tissue_pct
                    )
                    
                    # Save patch if requested
                    if self.save_patches and self.output_dir:
                        patch_dir = self.output_dir / slide_id
                        patch_dir.mkdir(exist_ok=True)
                        patch_path = patch_dir / f"{patch_info.patch_id}.png"
                        Image.fromarray(patch_array).save(patch_path)
                        
                    patches.append(patch_info)
                    
            self.logger.info(f"Extracted {len(patches)} patches from {slide_id}")
            
            return SlideData(
                slide_id=slide_id,
                patches=patches,
                metadata=metadata,
                thumbnail=thumbnail,
                tissue_mask=tissue_mask
            )
            
        finally:
            slide.close()
            
    def _calculate_tissue_percentage(self, patch: np.ndarray) -> float:
        """Calculate percentage of tissue in patch."""
        if patch.size == 0:
            return 0.0
            
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        tissue_mask = gray < self.background_threshold
        
        return tissue_mask.mean()
        
    def save_slide_data(self, slide_data: SlideData, output_path: Union[str, Path]):
        """
        Save processed slide data to HDF5 file.
        
        Args:
            slide_data: SlideData object to save
            output_path: Path to output HDF5 file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            # Save metadata
            metadata_group = f.create_group('metadata')
            for key, value in slide_data.metadata.items():
                if isinstance(value, (list, tuple)):
                    metadata_group.create_dataset(key, data=value)
                elif isinstance(value, dict):
                    subgroup = metadata_group.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str):
                            subgroup.attrs[subkey] = subvalue
                        else:
                            subgroup.create_dataset(subkey, data=subvalue)
                else:
                    metadata_group.attrs[key] = value
                    
            # Save thumbnail and tissue mask
            if slide_data.thumbnail is not None:
                f.create_dataset('thumbnail', data=slide_data.thumbnail)
            if slide_data.tissue_mask is not None:
                f.create_dataset('tissue_mask', data=slide_data.tissue_mask)
                
            # Save patch information
            patches_group = f.create_group('patches')
            for i, patch in enumerate(slide_data.patches):
                patch_group = patches_group.create_group(f'patch_{i:06d}')
                patch_group.attrs['x'] = patch.x
                patch_group.attrs['y'] = patch.y
                patch_group.attrs['level'] = patch.level
                patch_group.attrs['magnification'] = patch.magnification
                patch_group.attrs['patch_id'] = patch.patch_id
                patch_group.attrs['tissue_percentage'] = patch.tissue_percentage
                
                if patch.features is not None:
                    patch_group.create_dataset('features', data=patch.features)
                    
        self.logger.info(f"Saved slide data to {output_path}")
        
    @classmethod
    def load_slide_data(cls, input_path: Union[str, Path]) -> SlideData:
        """
        Load processed slide data from HDF5 file.
        
        Args:
            input_path: Path to HDF5 file
            
        Returns:
            SlideData object
        """
        input_path = Path(input_path)
        
        with h5py.File(input_path, 'r') as f:
            # Load metadata
            metadata = {}
            if 'metadata' in f:
                metadata_group = f['metadata']
                for key in metadata_group.attrs:
                    metadata[key] = metadata_group.attrs[key]
                for key in metadata_group.keys():
                    if isinstance(metadata_group[key], h5py.Group):
                        subdict = {}
                        for subkey in metadata_group[key].attrs:
                            subdict[subkey] = metadata_group[key].attrs[subkey]
                        for subkey in metadata_group[key].keys():
                            subdict[subkey] = metadata_group[key][subkey][()]
                        metadata[key] = subdict
                    else:
                        metadata[key] = metadata_group[key][()]
                        
            # Load thumbnail and tissue mask
            thumbnail = f['thumbnail'][()] if 'thumbnail' in f else None
            tissue_mask = f['tissue_mask'][()] if 'tissue_mask' in f else None
            
            # Load patches
            patches = []
            if 'patches' in f:
                patches_group = f['patches']
                for patch_key in sorted(patches_group.keys()):
                    patch_group = patches_group[patch_key]
                    
                    features = None
                    if 'features' in patch_group:
                        features = patch_group['features'][()]
                        
                    patch = PatchInfo(
                        x=patch_group.attrs['x'],
                        y=patch_group.attrs['y'],
                        level=patch_group.attrs['level'],
                        magnification=patch_group.attrs['magnification'],
                        patch_id=patch_group.attrs['patch_id'].decode('utf-8'),
                        tissue_percentage=patch_group.attrs['tissue_percentage'],
                        features=features
                    )
                    patches.append(patch)
                    
            # Extract slide_id from filename if not in metadata
            slide_id = metadata.get('slide_id', input_path.stem)
            
            return SlideData(
                slide_id=slide_id,
                patches=patches,
                metadata=metadata,
                thumbnail=thumbnail,
                tissue_mask=tissue_mask
            )