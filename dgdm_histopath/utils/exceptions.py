"""
Custom exception classes for DGDM Histopath Lab.

Provides structured error handling with detailed context for debugging
and monitoring in clinical and research environments.
"""

from typing import Any, Dict, Optional, List
import traceback
import sys
import logging
from datetime import datetime


class DGDMException(Exception):
    """Base exception class for DGDM-specific errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        severity: str = "ERROR"
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.suggestion = suggestion
        self.severity = severity
        self.timestamp = datetime.utcnow().isoformat()
        self.traceback_info = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to structured dictionary for logging."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "context": self.context,
            "suggestion": self.suggestion,
            "traceback": self.traceback_info
        }


class SlideProcessingError(DGDMException):
    """Errors related to whole-slide image processing."""
    pass


class ModelError(DGDMException):
    """Errors related to model operations."""
    pass


class DataValidationError(DGDMException):
    """Errors related to data validation and integrity."""
    pass


class ConfigurationError(DGDMException):
    """Errors related to configuration and setup."""
    pass


class QuantumError(DGDMException):
    """Errors related to quantum-enhanced components."""
    pass


class ClinicalValidationError(DGDMException):
    """Errors related to clinical validation and compliance."""
    pass


class SecurityError(DGDMException):
    """Security-related errors."""
    pass


class PerformanceError(DGDMException):
    """Performance-related errors and warnings."""
    pass


class ResourceError(DGDMException):
    """Resource availability and allocation errors."""
    pass


class ExceptionHandler:
    """Centralized exception handling with monitoring integration."""
    
    def __init__(self, logger_name: str = "dgdm_histopath"):
        self.logger = logging.getLogger(logger_name)
        self.error_counts = {}
        self.critical_errors = []
    
    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ) -> None:
        """Handle exception with logging and monitoring."""
        
        # Convert to DGDM exception if needed
        if not isinstance(exception, DGDMException):
            dgdm_exception = DGDMException(
                message=str(exception),
                error_code=type(exception).__name__,
                context=context,
                severity="ERROR"
            )
        else:
            dgdm_exception = exception
            if context:
                dgdm_exception.context.update(context)
        
        # Log the exception
        self._log_exception(dgdm_exception)
        
        # Update metrics
        self._update_metrics(dgdm_exception)
        
        # Handle critical errors
        if dgdm_exception.severity in ["CRITICAL", "FATAL"]:
            self._handle_critical_error(dgdm_exception)
        
        if reraise:
            raise dgdm_exception
    
    def _log_exception(self, exception: DGDMException) -> None:
        """Log exception with appropriate level."""
        log_data = exception.to_dict()
        
        if exception.severity == "WARNING":
            self.logger.warning(f"DGDM Warning: {exception.message}", extra=log_data)
        elif exception.severity == "ERROR":
            self.logger.error(f"DGDM Error: {exception.message}", extra=log_data)
        elif exception.severity in ["CRITICAL", "FATAL"]:
            self.logger.critical(f"DGDM Critical: {exception.message}", extra=log_data)
    
    def _update_metrics(self, exception: DGDMException) -> None:
        """Update error metrics for monitoring."""
        error_key = f"{exception.error_code}:{exception.severity}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
    
    def _handle_critical_error(self, exception: DGDMException) -> None:
        """Handle critical errors with immediate attention."""
        self.critical_errors.append(exception)
        
        # Send alerts (in real deployment, this would integrate with monitoring systems)
        self.logger.critical(
            f"CRITICAL ERROR DETECTED: {exception.message}",
            extra={"alert": True, "context": exception.context}
        )
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors for monitoring dashboards."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_breakdown": self.error_counts,
            "critical_count": len(self.critical_errors),
            "recent_critical": [e.to_dict() for e in self.critical_errors[-5:]]
        }


def safe_execute(func, *args, **kwargs):
    """Decorator for safe function execution with error handling."""
    def wrapper(*args, **kwargs):
        handler = ExceptionHandler()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            handler.handle_exception(e, context={
                "function": func.__name__,
                "args": str(args)[:200],  # Truncate long arguments
                "kwargs": str(kwargs)[:200]
            })
    return wrapper


def validate_input(
    value: Any,
    expected_type: type,
    name: str,
    additional_checks: Optional[List[callable]] = None
) -> None:
    """Validate input with detailed error messages."""
    
    # Type check
    if not isinstance(value, expected_type):
        raise DataValidationError(
            f"Invalid type for {name}: expected {expected_type.__name__}, got {type(value).__name__}",
            context={"parameter": name, "expected_type": expected_type.__name__, "actual_type": type(value).__name__},
            suggestion=f"Ensure {name} is of type {expected_type.__name__}"
        )
    
    # Additional checks
    if additional_checks:
        for check in additional_checks:
            try:
                if not check(value):
                    raise DataValidationError(
                        f"Validation failed for {name}",
                        context={"parameter": name, "check": check.__name__},
                        suggestion="Check the parameter constraints"
                    )
            except Exception as e:
                raise DataValidationError(
                    f"Validation error for {name}: {str(e)}",
                    context={"parameter": name, "validation_error": str(e)}
                )


def check_system_resources() -> Dict[str, Any]:
    """Check system resources and raise warnings/errors if insufficient."""
    import psutil
    
    resources = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    # Check critical resource levels
    if resources["memory_percent"] > 95:
        raise ResourceError(
            "Critical memory usage detected",
            context=resources,
            severity="CRITICAL",
            suggestion="Free up memory or scale to larger instance"
        )
    
    if resources["cpu_percent"] > 90:
        raise PerformanceError(
            "High CPU usage detected",
            context=resources,
            severity="WARNING",
            suggestion="Consider distributing workload or optimizing processing"
        )
    
    return resources


# Global exception handler instance
global_exception_handler = ExceptionHandler()