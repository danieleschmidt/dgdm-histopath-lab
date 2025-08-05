"""Enhanced configuration utilities with validation and security."""

import yaml
import json
import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional, List
from datetime import datetime
import copy
from contextlib import contextmanager


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass


class ConfigValidator:
    """Validate configuration against schemas and security policies."""
    
    SENSITIVE_KEYS = {
        'password', 'token', 'key', 'secret', 'credential',
        'api_key', 'auth_token', 'private_key'
    }
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> List[str]:
        """Validate file paths in configuration."""
        errors = []
        
        def check_path(key: str, value: Any, path_prefix: str = ""):
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            
            if isinstance(value, dict):
                for k, v in value.items():
                    check_path(k, v, current_path)
            elif isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
                if value and not Path(value).exists():
                    errors.append(f"Path does not exist: {current_path} = {value}")
                    
        for key, value in config.items():
            check_path(key, value)
            
        return errors
    
    @staticmethod
    def check_security(config: Dict[str, Any]) -> List[str]:
        """Check for security issues in configuration."""
        warnings = []
        
        def check_sensitive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for sensitive keys
                    if any(sensitive in key.lower() for sensitive in ConfigValidator.SENSITIVE_KEYS):
                        if isinstance(value, str) and len(value) < 20:
                            warnings.append(f"Potentially weak credential at {current_path}")
                    
                    check_sensitive(value, current_path)
                    
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_sensitive(item, f"{path}[{i}]")
                    
        check_sensitive(config)
        return warnings
    
    @staticmethod
    def validate_required_fields(config: Dict[str, Any], required_fields: List[str]) -> List[str]:
        """Validate that required fields are present."""
        errors = []
        
        for field in required_fields:
            if '.' in field:
                # Nested field
                parts = field.split('.')
                current = config
                try:
                    for part in parts:
                        current = current[part]
                except (KeyError, TypeError):
                    errors.append(f"Required field missing: {field}")
            else:
                if field not in config:
                    errors.append(f"Required field missing: {field}")
                    
        return errors


def load_config(
    config_path: Union[str, Path],
    validate: bool = True,
    required_fields: Optional[List[str]] = None,
    allow_environment_override: bool = True
) -> Dict[str, Any]:
    """Load and validate configuration from YAML or JSON file."""
    logger = logging.getLogger(__name__)
    
    try:
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
            
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration path is not a file: {config_path}")
            
        # Check file permissions
        if config_path.stat().st_mode & 0o077:
            logger.warning(f"Configuration file {config_path} has permissive permissions")
            
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    raise ConfigurationError(f"Invalid YAML format: {e}")
            elif config_path.suffix.lower() == '.json':
                try:
                    config = json.load(f)
                except json.JSONDecodeError as e:
                    raise ConfigurationError(f"Invalid JSON format: {e}")
            else:
                raise ConfigurationError(f"Unsupported config format: {config_path.suffix}")
                
        if config is None:
            raise ConfigurationError("Configuration file is empty or invalid")
            
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary/object")
            
        # Environment variable override
        if allow_environment_override:
            config = _apply_environment_overrides(config)
            
        # Validation
        if validate:
            validator = ConfigValidator()
            
            # Check required fields
            if required_fields:
                errors = validator.validate_required_fields(config, required_fields)
                if errors:
                    raise ConfigurationError(f"Validation errors: {'; '.join(errors)}")
                    
            # Check paths
            path_errors = validator.validate_paths(config)
            if path_errors:
                logger.warning(f"Path validation warnings: {'; '.join(path_errors)}")
                
            # Security check
            security_warnings = validator.check_security(config)
            if security_warnings:
                logger.warning(f"Security warnings: {'; '.join(security_warnings)}")
                
        # Log successful load
        config_hash = hashlib.md5(str(config).encode()).hexdigest()[:8]
        logger.info(f"Configuration loaded successfully from {config_path} (hash: {config_hash})")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def _apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""
    config = copy.deepcopy(config)
    
    # Look for environment variables with DGDM_ prefix
    for key, value in os.environ.items():
        if key.startswith('DGDM_'):
            config_key = key[5:].lower()  # Remove DGDM_ prefix
            
            # Convert to appropriate type
            if value.lower() in ('true', 'false'):
                config[config_key] = value.lower() == 'true'
            elif value.isdigit():
                config[config_key] = int(value)
            elif value.replace('.', '').isdigit():
                config[config_key] = float(value)
            else:
                config[config_key] = value
                
    return config


def save_config(
    config: Dict[str, Any],
    output_path: Union[str, Path],
    backup_existing: bool = True,
    secure_permissions: bool = True
):
    """Save configuration to YAML or JSON file with backup and security features."""
    logger = logging.getLogger(__name__)
    
    try:
        output_path = Path(output_path)
        
        # Create backup of existing file
        if backup_existing and output_path.exists():
            backup_path = output_path.with_suffix(f"{output_path.suffix}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            output_path.rename(backup_path)
            logger.info(f"Created backup: {backup_path}")
            
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write configuration
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=True)
            elif output_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2, sort_keys=True, ensure_ascii=False)
            else:
                raise ConfigurationError(f"Unsupported config format: {output_path.suffix}")
                
        # Set secure permissions
        if secure_permissions:
            output_path.chmod(0o600)  # Owner read/write only
            
        logger.info(f"Configuration saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {output_path}: {e}")
        raise


@contextmanager
def config_transaction(config_path: Union[str, Path]):
    """Context manager for safe configuration updates."""
    config_path = Path(config_path)
    temp_path = config_path.with_suffix(f"{config_path.suffix}.tmp")
    
    try:
        # Load original config
        original_config = load_config(config_path) if config_path.exists() else {}
        yield original_config
        
        # Save to temporary file first
        save_config(original_config, temp_path)
        
        # Atomic move
        temp_path.replace(config_path)
        
    except Exception:
        # Clean up temporary file on error
        if temp_path.exists():
            temp_path.unlink()
        raise


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configurations with deep merge."""
    def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result
    
    if not configs:
        return {}
        
    result = configs[0]
    for config in configs[1:]:
        result = deep_merge(result, config)
        
    return result


def validate_config_schema(config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    """Validate configuration against a schema."""
    errors = []
    
    def validate_field(config_value: Any, schema_value: Any, path: str = ""):
        if isinstance(schema_value, dict):
            if 'type' in schema_value:
                expected_type = schema_value['type']
                if expected_type == 'string' and not isinstance(config_value, str):
                    errors.append(f"{path}: expected string, got {type(config_value).__name__}")
                elif expected_type == 'integer' and not isinstance(config_value, int):
                    errors.append(f"{path}: expected integer, got {type(config_value).__name__}")
                elif expected_type == 'number' and not isinstance(config_value, (int, float)):
                    errors.append(f"{path}: expected number, got {type(config_value).__name__}")
                elif expected_type == 'boolean' and not isinstance(config_value, bool):
                    errors.append(f"{path}: expected boolean, got {type(config_value).__name__}")
                    
            if 'properties' in schema_value and isinstance(config_value, dict):
                for prop, prop_schema in schema_value['properties'].items():
                    prop_path = f"{path}.{prop}" if path else prop
                    if prop in config_value:
                        validate_field(config_value[prop], prop_schema, prop_path)
                    elif schema_value.get('required', []) and prop in schema_value['required']:
                        errors.append(f"{prop_path}: required field missing")
                        
    validate_field(config, schema)
    return errors