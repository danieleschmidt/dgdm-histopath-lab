"""
Advanced validation system for DGDM Histopath Lab.

Provides comprehensive data validation, integrity checks, and clinical compliance
validation for medical AI applications.
"""

import os
import re
import hashlib
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from PIL import Image
import logging
from datetime import datetime
import json

from dgdm_histopath.utils.exceptions import (
    DataValidationError, ClinicalValidationError, SecurityError,
    ConfigurationError, global_exception_handler
)


class BaseValidator:
    """Base validator class with common validation patterns."""
    
    def __init__(self, name: str, strict_mode: bool = False):
        self.name = name
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(f"dgdm_histopath.validators.{name}")
    
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Main validation method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _log_validation_error(self, message: str, context: Dict[str, Any]) -> None:
        """Log validation errors with context."""
        self.logger.error(f"Validation failed in {self.name}: {message}", extra=context)


class SlideValidator(BaseValidator):
    """Validator for whole-slide image files and metadata."""
    
    def __init__(self, strict_mode: bool = False):
        super().__init__("slide_validator", strict_mode)
        self.supported_formats = {'.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn'}
        self.min_file_size = 1024 * 1024  # 1MB minimum
        self.max_file_size = 50 * 1024 * 1024 * 1024  # 50GB maximum
    
    def validate(self, slide_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """Validate slide file and extract metadata."""
        slide_path = Path(slide_path)
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {},
            "security_check": True
        }
        
        try:
            # Check file existence
            if not slide_path.exists():
                raise DataValidationError(
                    f"Slide file not found: {slide_path}",
                    context={"slide_path": str(slide_path)},
                    suggestion="Verify the file path is correct"
                )
            
            # Check file format
            if slide_path.suffix.lower() not in self.supported_formats:
                error_msg = f"Unsupported file format: {slide_path.suffix}"
                if self.strict_mode:
                    raise DataValidationError(error_msg, context={"format": slide_path.suffix})
                else:
                    validation_result["warnings"].append(error_msg)
            
            # Check file size
            file_size = slide_path.stat().st_size
            validation_result["metadata"]["file_size"] = file_size
            
            if file_size < self.min_file_size:
                raise DataValidationError(
                    f"File too small: {file_size} bytes (minimum: {self.min_file_size})",
                    context={"file_size": file_size, "min_size": self.min_file_size}
                )
            
            if file_size > self.max_file_size:
                raise DataValidationError(
                    f"File too large: {file_size} bytes (maximum: {self.max_file_size})",
                    context={"file_size": file_size, "max_size": self.max_file_size}
                )
            
            # Security validation
            self._validate_file_security(slide_path, validation_result)
            
            # Extract basic metadata
            validation_result["metadata"].update(self._extract_metadata(slide_path))
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
            global_exception_handler.handle_exception(e, context={"slide_path": str(slide_path)}, reraise=False)
        
        return validation_result
    
    def _validate_file_security(self, slide_path: Path, result: Dict[str, Any]) -> None:
        """Validate file for security issues."""
        # Check for suspicious file patterns
        with open(slide_path, 'rb') as f:
            header = f.read(1024)
        
        # Check for executable signatures
        suspicious_patterns = [b'\x4D\x5A', b'\x7F\x45\x4C\x46']  # PE and ELF headers
        for pattern in suspicious_patterns:
            if pattern in header:
                result["security_check"] = False
                raise SecurityError(
                    "Suspicious file content detected",
                    context={"file_path": str(slide_path), "pattern": pattern.hex()},
                    severity="CRITICAL"
                )
    
    def _extract_metadata(self, slide_path: Path) -> Dict[str, Any]:
        """Extract metadata from slide file."""
        metadata = {
            "filename": slide_path.name,
            "extension": slide_path.suffix.lower(),
            "creation_time": datetime.fromtimestamp(slide_path.stat().st_ctime).isoformat(),
            "modification_time": datetime.fromtimestamp(slide_path.stat().st_mtime).isoformat(),
        }
        
        # Calculate file hash for integrity checking
        with open(slide_path, 'rb') as f:
            file_hash = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
            metadata["sha256"] = file_hash.hexdigest()
        
        return metadata


class ModelValidator(BaseValidator):
    """Validator for model configurations and checkpoints."""
    
    def __init__(self, strict_mode: bool = False):
        super().__init__("model_validator", strict_mode)
        self.required_model_keys = {'model_state_dict', 'optimizer_state_dict', 'epoch'}
        self.max_model_size = 10 * 1024 * 1024 * 1024  # 10GB maximum
    
    def validate(self, model_data: Union[Dict[str, Any], str, Path], **kwargs) -> Dict[str, Any]:
        """Validate model checkpoint or configuration."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Handle file path input
            if isinstance(model_data, (str, Path)):
                model_path = Path(model_data)
                if not model_path.exists():
                    raise DataValidationError(f"Model file not found: {model_path}")
                
                # Check file size
                file_size = model_path.stat().st_size
                if file_size > self.max_model_size:
                    raise DataValidationError(
                        f"Model file too large: {file_size} bytes",
                        context={"file_size": file_size, "max_size": self.max_model_size}
                    )
                
                # Load and validate checkpoint
                try:
                    checkpoint = torch.load(model_path, map_location='cpu')
                    validation_result["metadata"]["file_size"] = file_size
                    validation_result["metadata"]["file_path"] = str(model_path)
                except Exception as e:
                    raise DataValidationError(f"Failed to load model checkpoint: {str(e)}")
            
            elif isinstance(model_data, dict):
                checkpoint = model_data
            else:
                raise DataValidationError(f"Invalid model data type: {type(model_data)}")
            
            # Validate checkpoint structure
            self._validate_checkpoint_structure(checkpoint, validation_result)
            
            # Validate model architecture compatibility
            self._validate_model_architecture(checkpoint, validation_result)
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
            global_exception_handler.handle_exception(e, reraise=False)
        
        return validation_result
    
    def _validate_checkpoint_structure(self, checkpoint: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate checkpoint has required keys and structure."""
        missing_keys = self.required_model_keys - set(checkpoint.keys())
        if missing_keys:
            if self.strict_mode:
                raise DataValidationError(
                    f"Missing required keys in checkpoint: {missing_keys}",
                    context={"missing_keys": list(missing_keys)}
                )
            else:
                result["warnings"].append(f"Missing keys: {missing_keys}")
        
        # Validate metadata if present
        if 'metadata' in checkpoint:
            metadata = checkpoint['metadata']
            result["metadata"].update({
                "model_version": metadata.get('version', 'unknown'),
                "training_time": metadata.get('training_time'),
                "dataset": metadata.get('dataset'),
                "performance_metrics": metadata.get('metrics', {})
            })
    
    def _validate_model_architecture(self, checkpoint: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate model architecture parameters."""
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Check for reasonable parameter counts
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            result["metadata"]["total_parameters"] = total_params
            
            # Warn about extremely large models
            if total_params > 1e9:  # > 1B parameters
                result["warnings"].append(f"Very large model detected: {total_params:,} parameters")


class ClinicalValidator(BaseValidator):
    """Validator for clinical compliance and medical AI requirements."""
    
    def __init__(self, strict_mode: bool = True):  # Clinical validation should be strict by default
        super().__init__("clinical_validator", strict_mode)
        self.required_metadata = {
            'patient_id', 'study_date', 'institution', 'scanner_model',
            'magnification', 'pixel_spacing'
        }
        self.phi_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{2}/\d{2}/\d{4}\b',  # Date pattern
            r'\b[A-Za-z]+,\s*[A-Za-z]+\b'  # Name pattern
        ]
    
    def validate(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Validate clinical compliance and PHI protection."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "compliance_status": "PENDING",
            "phi_detected": False
        }
        
        try:
            # Validate required clinical metadata
            self._validate_clinical_metadata(data, validation_result)
            
            # Check for PHI (Protected Health Information)
            self._check_phi_presence(data, validation_result)
            
            # Validate data quality for clinical use
            self._validate_clinical_quality(data, validation_result)
            
            # Determine overall compliance status
            if not validation_result["errors"] and not validation_result["phi_detected"]:
                validation_result["compliance_status"] = "COMPLIANT"
            elif validation_result["errors"]:
                validation_result["compliance_status"] = "NON_COMPLIANT"
            else:
                validation_result["compliance_status"] = "REVIEW_REQUIRED"
                
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
            validation_result["compliance_status"] = "VALIDATION_FAILED"
            global_exception_handler.handle_exception(e, reraise=False)
        
        return validation_result
    
    def _validate_clinical_metadata(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate presence of required clinical metadata."""
        metadata = data.get('metadata', {})
        missing_fields = self.required_metadata - set(metadata.keys())
        
        if missing_fields:
            error_msg = f"Missing required clinical metadata: {missing_fields}"
            if self.strict_mode:
                raise ClinicalValidationError(
                    error_msg,
                    context={"missing_fields": list(missing_fields)},
                    suggestion="Ensure all required clinical metadata is provided"
                )
            else:
                result["warnings"].append(error_msg)
    
    def _check_phi_presence(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Check for potential PHI in the data."""
        data_str = json.dumps(data, default=str).lower()
        
        for pattern in self.phi_patterns:
            if re.search(pattern, data_str):
                result["phi_detected"] = True
                raise ClinicalValidationError(
                    "Potential PHI detected in data",
                    context={"pattern": pattern},
                    severity="CRITICAL",
                    suggestion="Remove or anonymize all PHI before processing"
                )
    
    def _validate_clinical_quality(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate data quality meets clinical standards."""
        metadata = data.get('metadata', {})
        
        # Check image quality parameters
        if 'magnification' in metadata:
            mag = metadata['magnification']
            if mag < 10 or mag > 100:
                result["warnings"].append(f"Unusual magnification: {mag}x")
        
        # Check for proper calibration
        if 'pixel_spacing' not in metadata:
            result["warnings"].append("Missing pixel spacing calibration")


class DataIntegrityValidator(BaseValidator):
    """Validator for data integrity and consistency checks."""
    
    def __init__(self, strict_mode: bool = False):
        super().__init__("integrity_validator", strict_mode)
    
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate data integrity and consistency."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "integrity_checks": {}
        }
        
        try:
            if isinstance(data, np.ndarray):
                self._validate_array_integrity(data, validation_result)
            elif isinstance(data, torch.Tensor):
                self._validate_tensor_integrity(data, validation_result)
            elif isinstance(data, dict):
                self._validate_dict_integrity(data, validation_result)
            elif isinstance(data, (list, tuple)):
                self._validate_sequence_integrity(data, validation_result)
                
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(str(e))
            global_exception_handler.handle_exception(e, reraise=False)
        
        return validation_result
    
    def _validate_array_integrity(self, data: np.ndarray, result: Dict[str, Any]) -> None:
        """Validate numpy array integrity."""
        checks = {}
        
        # Check for NaN/Inf values
        checks["has_nan"] = np.isnan(data).any()
        checks["has_inf"] = np.isinf(data).any()
        
        if checks["has_nan"] or checks["has_inf"]:
            error_msg = "Array contains NaN or Inf values"
            if self.strict_mode:
                raise DataValidationError(error_msg, context=checks)
            else:
                result["warnings"].append(error_msg)
        
        # Check data range
        checks["min_value"] = float(data.min())
        checks["max_value"] = float(data.max())
        checks["mean_value"] = float(data.mean())
        checks["std_value"] = float(data.std())
        
        result["integrity_checks"]["array"] = checks
    
    def _validate_tensor_integrity(self, data: torch.Tensor, result: Dict[str, Any]) -> None:
        """Validate PyTorch tensor integrity."""
        checks = {}
        
        # Check for NaN/Inf values
        checks["has_nan"] = torch.isnan(data).any().item()
        checks["has_inf"] = torch.isinf(data).any().item()
        
        if checks["has_nan"] or checks["has_inf"]:
            error_msg = "Tensor contains NaN or Inf values"
            if self.strict_mode:
                raise DataValidationError(error_msg, context=checks)
            else:
                result["warnings"].append(error_msg)
        
        # Check tensor properties
        checks["shape"] = list(data.shape)
        checks["dtype"] = str(data.dtype)
        checks["device"] = str(data.device)
        checks["requires_grad"] = data.requires_grad
        
        result["integrity_checks"]["tensor"] = checks
    
    def _validate_dict_integrity(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Validate dictionary data integrity."""
        checks = {
            "key_count": len(data),
            "empty_values": sum(1 for v in data.values() if v is None or v == ""),
            "nested_depth": self._get_dict_depth(data)
        }
        
        # Check for excessively nested structures
        if checks["nested_depth"] > 10:
            result["warnings"].append(f"Deeply nested dictionary: {checks['nested_depth']} levels")
        
        result["integrity_checks"]["dict"] = checks
    
    def _validate_sequence_integrity(self, data: Union[List, Tuple], result: Dict[str, Any]) -> None:
        """Validate sequence data integrity."""
        checks = {
            "length": len(data),
            "empty_count": sum(1 for item in data if not item),
            "type_consistency": len(set(type(item).__name__ for item in data)) == 1 if data else True
        }
        
        if not checks["type_consistency"]:
            result["warnings"].append("Inconsistent types in sequence")
        
        result["integrity_checks"]["sequence"] = checks
    
    def _get_dict_depth(self, d: Dict[str, Any], depth: int = 0) -> int:
        """Calculate maximum nesting depth of dictionary."""
        if not isinstance(d, dict):
            return depth
        return max(self._get_dict_depth(v, depth + 1) for v in d.values()) if d else depth


class ValidationPipeline:
    """Orchestrates multiple validators in a pipeline."""
    
    def __init__(self, validators: List[BaseValidator], fail_fast: bool = False):
        self.validators = validators
        self.fail_fast = fail_fast
        self.logger = logging.getLogger("dgdm_histopath.validation_pipeline")
    
    def validate_all(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Run all validators in the pipeline."""
        pipeline_result = {
            "overall_valid": True,
            "validator_results": {},
            "summary": {
                "total_validators": len(self.validators),
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }
        
        for validator in self.validators:
            try:
                result = validator.validate(data, **kwargs)
                pipeline_result["validator_results"][validator.name] = result
                
                if result["valid"]:
                    pipeline_result["summary"]["passed"] += 1
                else:
                    pipeline_result["summary"]["failed"] += 1
                    pipeline_result["overall_valid"] = False
                    
                    if self.fail_fast:
                        break
                
                pipeline_result["summary"]["warnings"] += len(result.get("warnings", []))
                
            except Exception as e:
                pipeline_result["validator_results"][validator.name] = {
                    "valid": False,
                    "errors": [str(e)],
                    "exception": type(e).__name__
                }
                pipeline_result["summary"]["failed"] += 1
                pipeline_result["overall_valid"] = False
                
                if self.fail_fast:
                    break
        
        return pipeline_result


# Pre-configured validation pipelines for common use cases
def get_slide_validation_pipeline(strict_mode: bool = False) -> ValidationPipeline:
    """Get validation pipeline for slide processing."""
    return ValidationPipeline([
        SlideValidator(strict_mode=strict_mode),
        DataIntegrityValidator(strict_mode=strict_mode)
    ])


def get_clinical_validation_pipeline(strict_mode: bool = True) -> ValidationPipeline:
    """Get validation pipeline for clinical deployment."""
    return ValidationPipeline([
        SlideValidator(strict_mode=strict_mode),
        ClinicalValidator(strict_mode=strict_mode),
        DataIntegrityValidator(strict_mode=strict_mode)
    ], fail_fast=True)


def get_model_validation_pipeline(strict_mode: bool = False) -> ValidationPipeline:
    """Get validation pipeline for model validation."""
    return ValidationPipeline([
        ModelValidator(strict_mode=strict_mode),
        DataIntegrityValidator(strict_mode=strict_mode)
    ])