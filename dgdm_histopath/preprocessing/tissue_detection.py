"""
Tissue detection and segmentation for histopathology slides.

Implements automated tissue detection using morphological operations,
color-based thresholding, and machine learning approaches.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union
from sklearn.cluster import KMeans
import logging


class TissueDetector:
    """
    Automated tissue detection for whole-slide images.
    
    Uses a combination of color thresholding, morphological operations,
    and optional machine learning clustering to identify tissue regions.
    """
    
    def __init__(
        self,
        background_threshold: int = 220,
        min_area: int = 1000,
        gaussian_blur_kernel: int = 5,
        morphology_kernel: int = 5,
        use_clustering: bool = False,
        n_clusters: int = 3
    ):
        """
        Initialize tissue detector.
        
        Args:
            background_threshold: Intensity threshold for background (0-255)
            min_area: Minimum area for tissue regions (pixels)
            gaussian_blur_kernel: Kernel size for Gaussian blur
            morphology_kernel: Kernel size for morphological operations
            use_clustering: Whether to use K-means clustering
            n_clusters: Number of clusters for K-means
        """
        self.background_threshold = background_threshold
        self.min_area = min_area
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.morphology_kernel = morphology_kernel
        self.use_clustering = use_clustering
        self.n_clusters = n_clusters
        
        self.logger = logging.getLogger(__name__)
        
    def detect_tissue(self, image: np.ndarray) -> np.ndarray:
        """
        Detect tissue regions in an RGB image.
        
        Args:
            image: RGB image array [H, W, 3]
            
        Returns:
            Binary tissue mask [H, W]
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB (H, W, 3)")
            
        # Apply Gaussian blur to reduce noise
        if self.gaussian_blur_kernel > 0:
            image_blur = cv2.GaussianBlur(
                image, 
                (self.gaussian_blur_kernel, self.gaussian_blur_kernel), 
                0
            )
        else:
            image_blur = image.copy()
            
        if self.use_clustering:
            mask = self._detect_tissue_clustering(image_blur)
        else:
            mask = self._detect_tissue_threshold(image_blur)
            
        # Apply morphological operations
        mask = self._apply_morphology(mask)
        
        # Remove small components
        mask = self._remove_small_objects(mask)
        
        return mask.astype(np.uint8)
        
    def _detect_tissue_threshold(self, image: np.ndarray) -> np.ndarray:
        """Detect tissue using color thresholding."""
        # Convert to different color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Multiple thresholding approaches
        # 1. Simple intensity threshold
        mask_intensity = gray < self.background_threshold
        
        # 2. Otsu's thresholding
        _, mask_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask_otsu = mask_otsu > 0
        
        # 3. HSV-based thresholding (avoid white/very light regions)
        mask_hsv = (hsv[:, :, 1] > 20) & (hsv[:, :, 2] < 240)
        
        # Combine masks
        combined_mask = mask_intensity & mask_otsu & mask_hsv
        
        return combined_mask
        
    def _detect_tissue_clustering(self, image: np.ndarray) -> np.ndarray:
        """Detect tissue using K-means clustering."""
        # Reshape image for clustering
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Find cluster centers
        centers = kmeans.cluster_centers_
        
        # Identify background cluster (highest intensity)
        background_cluster = np.argmax(np.mean(centers, axis=1))
        
        # Create tissue mask (non-background clusters)
        tissue_mask = labels != background_cluster
        tissue_mask = tissue_mask.reshape(h, w)
        
        return tissue_mask
        
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean up mask."""
        if self.morphology_kernel <= 0:
            return mask
            
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel, self.morphology_kernel)
        )
        
        # Close small holes
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask > 0
        
    def _remove_small_objects(self, mask: np.ndarray) -> np.ndarray:
        """Remove small connected components."""
        if self.min_area <= 0:
            return mask
            
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        # Create new mask keeping only large components
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                new_mask[labels == i] = True
                
        return new_mask
        
    def get_tissue_statistics(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """
        Compute statistics about detected tissue regions.
        
        Args:
            image: Original RGB image
            mask: Binary tissue mask
            
        Returns:
            Dictionary with tissue statistics
        """
        total_pixels = mask.size
        tissue_pixels = np.sum(mask)
        
        stats = {
            'total_pixels': total_pixels,
            'tissue_pixels': tissue_pixels,
            'tissue_percentage': tissue_pixels / total_pixels * 100,
            'background_pixels': total_pixels - tissue_pixels,
            'num_components': 0,
            'largest_component_area': 0,
            'mean_component_area': 0
        }
        
        # Analyze connected components
        num_labels, labels, component_stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels > 1:  # At least one tissue component
            component_areas = component_stats[1:, cv2.CC_STAT_AREA]  # Skip background
            stats['num_components'] = len(component_areas)
            stats['largest_component_area'] = np.max(component_areas)
            stats['mean_component_area'] = np.mean(component_areas)
            
        return stats