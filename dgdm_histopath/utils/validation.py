"""Input validation and sanitization utilities."""

import re
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import hashlib
import mimetypes


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    # Malicious patterns to detect
    SUSPICIOUS_PATTERNS = [
        r'\.\./',           # Directory traversal
        r'<script.*?>',     # XSS attempts
        r'javascript:',     # JavaScript protocol
        r'vbscript:',       # VBScript protocol
        r'data:text/html',  # Data URI HTML
        r'<\s*iframe',      # Iframe injection
        r'union\s+select',  # SQL injection
        r'exec\s*\(',       # Code execution
        r'eval\s*\(',       # Code evaluation
        r'__import__',      # Python imports
        r'subprocess',      # Process execution
        r'os\.system',      # System commands
    ]
    
    ALLOWED_FILE_EXTENSIONS = {
        '.svs', '.ndpi', '.mrxs', '.scn', '.vms', '.vmu', '.dcm',  # Medical images
        '.tiff', '.tif', '.png', '.jpg', '.jpeg',                 # Standard images
        '.yaml', '.yml', '.json', '.csv', '.txt',                 # Config/data
        '.h5', '.hdf5', '.npz', '.npy',                          # Data formats
        '.ckpt', '.pth', '.pt', '.safetensors'                   # Model formats
    }
    
    MAX_FILE_SIZE = 50 * 1024 * 1024 * 1024  # 50GB for medical images
    MAX_STRING_LENGTH = 10000
    
    @staticmethod
    def sanitize_string(value: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input for security."""
        if not isinstance(value, str):
            raise ValidationError(f"Expected string, got {type(value).__name__}")
            
        # Check length
        max_len = max_length or InputValidator.MAX_STRING_LENGTH
        if len(value) > max_len:
            raise ValidationError(f"String too long: {len(value)} > {max_len}")
            
        # Check for suspicious patterns
        for pattern in InputValidator.SUSPICIOUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityError(f"Suspicious pattern detected: {pattern}")
                
        # Basic sanitization
        sanitized = value.strip()
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        return sanitized
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path], check_exists: bool = True) -> Path:
        """Validate and sanitize file path."""
        try:
            # Convert to Path object
            path = Path(file_path)
            
            # Basic path validation
            if not str(path):
                raise ValidationError("Empty file path")
                
            # Check for path traversal
            resolved_path = path.resolve()
            if '..' in str(path) or str(resolved_path) != str(path.resolve()):
                raise SecurityError(f"Path traversal detected: {path}")
                
            # Check existence if required
            if check_exists and not resolved_path.exists():
                raise ValidationError(f"File does not exist: {resolved_path}")
                
            # Check if it's actually a file when required
            if check_exists and not resolved_path.is_file():
                raise ValidationError(f"Path is not a file: {resolved_path}")
                
            # Validate file extension
            if resolved_path.suffix.lower() not in InputValidator.ALLOWED_FILE_EXTENSIONS:
                raise SecurityError(f"File extension not allowed: {resolved_path.suffix}")
                
            # Check file size
            if check_exists and resolved_path.stat().st_size > InputValidator.MAX_FILE_SIZE:
                raise ValidationError(f"File too large: {resolved_path.stat().st_size} bytes")
                
            return resolved_path
            
        except (OSError, PermissionError) as e:
            raise ValidationError(f"File system error: {e}")
    
    @staticmethod
    def validate_directory_path(dir_path: Union[str, Path], create_if_missing: bool = False) -> Path:
        """Validate directory path."""
        try:
            path = Path(dir_path)
            resolved_path = path.resolve()
            
            # Check for path traversal
            if '..' in str(path):
                raise SecurityError(f"Path traversal detected: {path}")
                
            # Create if requested
            if create_if_missing:
                resolved_path.mkdir(parents=True, exist_ok=True)
            elif not resolved_path.exists():
                raise ValidationError(f"Directory does not exist: {resolved_path}")
                
            if resolved_path.exists() and not resolved_path.is_dir():
                raise ValidationError(f"Path is not a directory: {resolved_path}")
                
            return resolved_path
            
        except (OSError, PermissionError) as e:
            raise ValidationError(f"Directory system error: {e}")
    
    @staticmethod
    def validate_numeric(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Validate numeric input."""
        try:
            num_val = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid numeric value: {value}")
            
        if min_val is not None and num_val < min_val:
            raise ValidationError(f"Value too small: {num_val} < {min_val}")
            
        if max_val is not None and num_val > max_val:
            raise ValidationError(f"Value too large: {num_val} > {max_val}")
            
        return num_val
    
    @staticmethod
    def validate_integer(value: Any, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
        """Validate integer input."""
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid integer value: {value}")
            
        if min_val is not None and int_val < min_val:
            raise ValidationError(f"Value too small: {int_val} < {min_val}")
            
        if max_val is not None and int_val > max_val:
            raise ValidationError(f"Value too large: {int_val} > {max_val}")
            
        return int_val
    
    @staticmethod
    def validate_boolean(value: Any) -> bool:
        """Validate boolean input."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
            else:
                raise ValidationError(f"Invalid boolean value: {value}")
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            raise ValidationError(f"Cannot convert to boolean: {type(value).__name__}")
    
    @staticmethod
    def validate_enum(value: Any, allowed_values: List[Any]) -> Any:
        """Validate enum/choice input."""
        if value not in allowed_values:
            raise ValidationError(f"Invalid choice: {value}. Allowed: {allowed_values}")
        return value
    
    @staticmethod
    def validate_positive_number(value: Any, field_name: str) -> float:
        """Validate that a value is a positive number."""
        try:
            num_val = float(value)
            if num_val <= 0:
                raise ValidationError(f"{field_name} must be positive, got {num_val}")
            return num_val
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid numeric value for {field_name}: {value}")
    
    @staticmethod
    def validate_range(value: Any, min_val: float, max_val: float, field_name: str) -> float:
        """Validate that a value is within a specified range."""
        try:
            num_val = float(value)
            if not (min_val <= num_val <= max_val):
                raise ValidationError(f"{field_name} must be between {min_val} and {max_val}, got {num_val}")
            return num_val
        except (ValueError, TypeError):
            raise ValidationError(f"Invalid numeric value for {field_name}: {value}")
    
    @staticmethod
    def validate_quantum_state(state: Any) -> Any:
        """Validate quantum state array."""
        import numpy as np
        
        if not isinstance(state, np.ndarray):
            raise ValidationError("Quantum state must be numpy array")
        
        if not np.iscomplexobj(state):
            raise ValidationError("Quantum state must have complex dtype")
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(state)):
            raise ValidationError("Quantum state contains NaN or infinite values")
        
        # Check normalization
        norm = np.linalg.norm(state)
        if abs(norm - 1.0) > 1e-6:
            raise ValidationError(f"Quantum state not normalized: norm = {norm}")
        
        return state
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration."""
        validated = {}
        
        # Required fields
        required_fields = ['node_features', 'hidden_dims']
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"Required field missing: {field}")
                
        # Validate node_features
        validated['node_features'] = InputValidator.validate_integer(
            config['node_features'], min_val=1, max_val=10000
        )
        
        # Validate hidden_dims
        if not isinstance(config['hidden_dims'], list):
            raise ValidationError("hidden_dims must be a list")
        validated['hidden_dims'] = [
            InputValidator.validate_integer(dim, min_val=1, max_val=10000)
            for dim in config['hidden_dims']
        ]
        
        # Optional fields with validation
        if 'num_diffusion_steps' in config:
            validated['num_diffusion_steps'] = InputValidator.validate_integer(
                config['num_diffusion_steps'], min_val=1, max_val=1000
            )
            
        if 'attention_heads' in config:
            validated['attention_heads'] = InputValidator.validate_integer(
                config['attention_heads'], min_val=1, max_val=32
            )
            
        if 'dropout' in config:
            validated['dropout'] = InputValidator.validate_numeric(
                config['dropout'], min_val=0.0, max_val=0.9
            )
            
        if 'learning_rate' in config:
            validated['learning_rate'] = InputValidator.validate_numeric(
                config['learning_rate'], min_val=1e-6, max_val=1.0
            )
            
        return validated


class FileValidator:
    """File-specific validation utilities."""
    
    @staticmethod
    def validate_medical_image(file_path: Path) -> Dict[str, Any]:
        """Validate medical image file."""
        logger = logging.getLogger(__name__)
        
        try:
            # Check file signature/magic numbers
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                
            # Basic format validation
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            info = {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'mime_type': mime_type,
                'extension': file_path.suffix.lower(),
                'is_valid': True,
                'warnings': []
            }
            
            # Size validation
            if info['file_size'] == 0:
                raise ValidationError("File is empty")
            elif info['file_size'] > InputValidator.MAX_FILE_SIZE:
                raise ValidationError(f"File too large: {info['file_size']} bytes")
                
            # Extension-specific validation
            if file_path.suffix.lower() in ['.svs', '.ndpi', '.mrxs']:
                # Validate whole slide image formats
                info['format'] = 'whole_slide_image'
                
                # Check for OpenSlide compatibility (if available)
                try:
                    import openslide
                    slide = openslide.OpenSlide(str(file_path))
                    info['dimensions'] = slide.dimensions
                    info['levels'] = slide.level_count
                    info['magnification'] = slide.properties.get('openslide.objective-power', 'unknown')
                    slide.close()
                except ImportError:
                    info['warnings'].append("OpenSlide not available for detailed validation")
                except Exception as e:
                    info['warnings'].append(f"OpenSlide validation failed: {e}")
                    
            elif file_path.suffix.lower() in ['.tiff', '.tif', '.png', '.jpg', '.jpeg']:
                info['format'] = 'standard_image'
                
                # Basic image validation with PIL
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        info['dimensions'] = img.size
                        info['mode'] = img.mode
                        info['format_details'] = img.format
                except ImportError:
                    info['warnings'].append("PIL not available for image validation")
                except Exception as e:
                    info['warnings'].append(f"Image validation failed: {e}")
                    
            return info
            
        except Exception as e:
            logger.error(f"File validation failed for {file_path}: {e}")
            raise ValidationError(f"File validation failed: {e}")
    
    @staticmethod
    def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
        """Compute file hash for integrity checking."""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
                
        return hash_obj.hexdigest()


def validate_tensor_shape(tensor_shape: Tuple[int, ...], expected_dims: int, min_size: int = 1) -> Tuple[int, ...]:
    """Validate tensor shape."""
    if not isinstance(tensor_shape, (tuple, list)):
        raise ValidationError(f"Invalid tensor shape type: {type(tensor_shape)}")
        
    if len(tensor_shape) != expected_dims:
        raise ValidationError(f"Expected {expected_dims} dimensions, got {len(tensor_shape)}")
        
    for i, dim in enumerate(tensor_shape):
        if not isinstance(dim, int) or dim < min_size:
            raise ValidationError(f"Invalid dimension {i}: {dim} (must be >= {min_size})")
            
    return tuple(tensor_shape)


def validate_gpu_availability() -> Dict[str, Any]:
    """Validate GPU availability and resources."""
    gpu_info = {
        'available': False,
        'device_count': 0,
        'devices': [],
        'warnings': []
    }
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            
            for i in range(gpu_info['device_count']):
                device_props = torch.cuda.get_device_properties(i)
                gpu_info['devices'].append({
                    'index': i,
                    'name': device_props.name,
                    'memory_total': device_props.total_memory,
                    'memory_free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i),
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
                
    except ImportError:
        gpu_info['warnings'].append("PyTorch not available")
    except Exception as e:
        gpu_info['warnings'].append(f"GPU detection failed: {e}")
        
    return gpu_info