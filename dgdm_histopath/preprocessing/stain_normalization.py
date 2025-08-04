"""
Stain normalization for histopathology images.

Implements Macenko and Reinhard stain normalization methods
to reduce color variations between different scanners and staining protocols.
"""

import numpy as np
import cv2
from typing import Optional, Tuple
import logging


class StainNormalizer:
    """
    Histopathology stain normalization using Macenko method.
    
    Normalizes H&E stained images to a reference template to reduce
    color variations caused by different scanners and staining protocols.
    """
    
    def __init__(
        self,
        method: str = "macenko",
        target_concentrations: Optional[np.ndarray] = None,
        target_stains: Optional[np.ndarray] = None,
        io_threshold: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.15
    ):
        """
        Initialize stain normalizer.
        
        Args:
            method: Normalization method ("macenko" or "reinhard")
            target_concentrations: Target stain concentrations [2]
            target_stains: Target stain vectors [2, 3]
            io_threshold: Optical density threshold
            alpha: Percentile for robust statistics
            beta: Transparency threshold
        """
        self.method = method.lower()
        self.io_threshold = io_threshold
        self.alpha = alpha
        self.beta = beta
        
        # Default H&E stain vectors (from Macenko et al.)
        if target_stains is None:
            self.target_stains = np.array([
                [0.5626, 0.2159, 0.7201],  # Hematoxylin
                [0.6500, 0.7044, 0.2864]   # Eosin
            ])
        else:
            self.target_stains = target_stains
            
        # Default target concentrations
        if target_concentrations is None:
            self.target_concentrations = np.array([1.9705, 1.0308])
        else:
            self.target_concentrations = target_concentrations
            
        self.logger = logging.getLogger(__name__)
        
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize staining of an H&E image.
        
        Args:
            image: RGB image array [H, W, 3] with values in [0, 255]
            
        Returns:
            Normalized RGB image [H, W, 3]
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB (H, W, 3)")
            
        if self.method == "macenko":
            return self._normalize_macenko(image)
        elif self.method == "reinhard":
            return self._normalize_reinhard(image)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
    def _normalize_macenko(self, image: np.ndarray) -> np.ndarray:
        """Macenko stain normalization."""
        # Convert to optical density
        od = self._rgb_to_od(image)
        
        # Remove transparent pixels
        od_flat = od.reshape(-1, 3)
        mask = np.sum(od_flat, axis=1) > self.io_threshold
        od_filtered = od_flat[mask]
        
        if od_filtered.shape[0] == 0:
            return image  # Return original if no tissue found
            
        # Compute eigenvectors
        cov_matrix = np.cov(od_filtered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        
        # Project data onto first two eigenvectors
        proj_data = od_filtered @ eigenvecs[:, :2]
        
        # Find robust extremes
        phi = np.arctan2(proj_data[:, 1], proj_data[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        
        # Get stain vectors
        v1 = eigenvecs[:, :2] @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = eigenvecs[:, :2] @ np.array([np.cos(max_phi), np.sin(max_phi)])
        
        # Ensure correct sign orientation
        if v1[0] < 0:
            v1 *= -1
        if v2[0] < 0:
            v2 *= -1
            
        source_stains = np.array([v1, v2])
        
        # Compute concentrations
        concentrations = self._get_concentrations(od, source_stains)
        
        # Normalize concentrations
        max_concs = np.percentile(concentrations, 99, axis=0)
        source_concentrations = max_concs
        
        # Apply normalization
        normalized_od = self._apply_normalization(
            concentrations, source_stains, source_concentrations
        )
        
        # Convert back to RGB
        normalized_rgb = self._od_to_rgb(normalized_od)
        
        return np.clip(normalized_rgb, 0, 255).astype(np.uint8)
        
    def _normalize_reinhard(self, image: np.ndarray) -> np.ndarray:
        """Reinhard color normalization in LAB space."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Compute mean and std for each channel
        source_mean = np.mean(lab, axis=(0, 1))
        source_std = np.std(lab, axis=(0, 1))
        
        # Target statistics (computed from reference image)
        target_mean = np.array([74.46, 10.89, 5.46])  # Example values
        target_std = np.array([18.32, 8.67, 4.21])
        
        # Normalize
        lab_norm = lab.copy()
        for i in range(3):
            lab_norm[:, :, i] = (lab[:, :, i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
            
        # Convert back to RGB
        rgb_norm = cv2.cvtColor(lab_norm.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return rgb_norm
        
    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density."""
        rgb = rgb.astype(np.float32) + 1e-6  # Avoid log(0)
        rgb /= 255.0
        od = -np.log(rgb)
        return od
        
    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density to RGB."""
        rgb = np.exp(-od) * 255.0
        return rgb
        
    def _get_concentrations(self, od: np.ndarray, stains: np.ndarray) -> np.ndarray:
        """Get stain concentrations using least squares."""
        od_flat = od.reshape(-1, 3)
        concentrations = np.linalg.lstsq(stains.T, od_flat.T, rcond=None)[0].T
        concentrations = np.maximum(concentrations, 0)  # Non-negative
        return concentrations.reshape(od.shape[:2] + (2,))
        
    def _apply_normalization(
        self, 
        concentrations: np.ndarray, 
        source_stains: np.ndarray,
        source_concentrations: np.ndarray
    ) -> np.ndarray:
        """Apply stain normalization transformation."""
        # Normalize concentrations
        conc_norm = concentrations.copy()
        for i in range(2):
            conc_norm[:, :, i] *= (self.target_concentrations[i] / source_concentrations[i])
            
        # Reconstruct optical density
        h, w = concentrations.shape[:2]
        conc_flat = conc_norm.reshape(-1, 2)
        od_flat = conc_flat @ self.target_stains
        od_norm = od_flat.reshape(h, w, 3)
        
        return od_norm
        
    def fit_to_template(self, template_image: np.ndarray):
        """
        Fit normalizer parameters to a template image.
        
        Args:
            template_image: Template RGB image [H, W, 3]
        """
        if self.method != "macenko":
            self.logger.warning("Template fitting only supported for Macenko method")
            return
            
        # Extract stain vectors and concentrations from template
        od = self._rgb_to_od(template_image)
        
        # Remove transparent pixels
        od_flat = od.reshape(-1, 3)
        mask = np.sum(od_flat, axis=1) > self.io_threshold
        od_filtered = od_flat[mask]
        
        if od_filtered.shape[0] == 0:
            self.logger.warning("No tissue found in template image")
            return
            
        # Compute eigenvectors
        cov_matrix = np.cov(od_filtered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        
        # Project data onto first two eigenvectors
        proj_data = od_filtered @ eigenvecs[:, :2]
        
        # Find robust extremes
        phi = np.arctan2(proj_data[:, 1], proj_data[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)
        
        # Get stain vectors
        v1 = eigenvecs[:, :2] @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = eigenvecs[:, :2] @ np.array([np.cos(max_phi), np.sin(max_phi)])
        
        # Ensure correct sign orientation
        if v1[0] < 0:
            v1 *= -1
        if v2[0] < 0:
            v2 *= -1
            
        self.target_stains = np.array([v1, v2])
        
        # Compute target concentrations
        concentrations = self._get_concentrations(od, self.target_stains)
        self.target_concentrations = np.percentile(concentrations, 99, axis=0)
        
        self.logger.info("Fitted normalizer to template image")