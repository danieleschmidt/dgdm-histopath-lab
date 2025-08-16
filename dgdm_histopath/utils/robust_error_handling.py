"""
Robust Error Handling System for DGDM Histopath Lab.

Comprehensive error handling, recovery mechanisms, and graceful degradation
for production-ready histopathology AI systems.
"""

import traceback
import functools
import inspect
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, TypeVar
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from pathlib import Path

# Type variable for decorated functions
F = TypeVar('F', bound=Callable[..., Any])


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    INPUT_VALIDATION = "input_validation"
    MODEL_INFERENCE = "model_inference"
    DATA_PROCESSING = "data_processing"
    MEMORY_OVERFLOW = "memory_overflow"
    NETWORK_FAILURE = "network_failure"
    FILE_SYSTEM = "file_system"
    DEPENDENCY_ERROR = "dependency_error"
    CONFIGURATION = "configuration"
    QUANTUM_FAILURE = "quantum_failure"
    CLINICAL_SAFETY = "clinical_safety"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    function_name: str
    module_name: str
    arguments: Dict[str, Any]
    timestamp: float
    severity: ErrorSeverity
    category: ErrorCategory
    patient_id: Optional[str] = None
    slide_id: Optional[str] = None
    model_version: Optional[str] = None
    retry_count: int = 0
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'function_name': self.function_name,
            'module_name': self.module_name,
            'arguments': str(self.arguments),
            'timestamp': self.timestamp,
            'severity': self.severity.value,
            'category': self.category.value,
            'patient_id': self.patient_id,
            'slide_id': self.slide_id,
            'model_version': self.model_version,
            'retry_count': self.retry_count,
            'stack_trace': self.stack_trace
        }


class RobustErrorHandler:
    """
    Comprehensive error handling system with recovery mechanisms.
    
    Features:
    - Automatic retry with exponential backoff
    - Graceful degradation strategies
    - Clinical safety checks
    - Memory management
    - Error reporting and analytics
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        enable_graceful_degradation: bool = True,
        log_all_errors: bool = True,
        error_log_path: Optional[Path] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enable_graceful_degradation = enable_graceful_degradation
        self.log_all_errors = log_all_errors
        
        # Setup error logging
        self.error_log_path = error_log_path or Path("logs/error_log.json")
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'by_category': {},
            'by_severity': {},
            'recovery_success_rate': 0.0
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.MEMORY_OVERFLOW: self._handle_memory_overflow,
            ErrorCategory.MODEL_INFERENCE: self._handle_model_failure,
            ErrorCategory.DATA_PROCESSING: self._handle_data_processing_error,
            ErrorCategory.NETWORK_FAILURE: self._handle_network_failure,
            ErrorCategory.CLINICAL_SAFETY: self._handle_clinical_safety_error
        }
        
        self.logger = logging.getLogger(__name__)

    def robust_execution(
        self,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.DATA_PROCESSING,
        max_retries: Optional[int] = None,
        enable_recovery: bool = True,
        fallback_value: Any = None
    ):
        """
        Decorator for robust function execution with error handling.
        
        Args:
            severity: Error severity level
            category: Error category for specialized handling
            max_retries: Override default retry count
            enable_recovery: Enable automatic recovery
            fallback_value: Value to return if all recovery fails
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                context = ErrorContext(
                    function_name=func.__name__,
                    module_name=func.__module__,
                    arguments={'args': args, 'kwargs': kwargs},
                    timestamp=time.time(),
                    severity=severity,
                    category=category
                )
                
                # Extract clinical context if available
                context.patient_id = kwargs.get('patient_id')
                context.slide_id = kwargs.get('slide_id')
                context.model_version = kwargs.get('model_version')
                
                retries = max_retries or self.max_retries
                last_exception = None
                
                for attempt in range(retries + 1):
                    try:
                        # Memory check before execution
                        if category == ErrorCategory.MODEL_INFERENCE:
                            self._check_memory_usage()
                        
                        result = func(*args, **kwargs)
                        
                        # Success - reset retry count and return
                        if attempt > 0:
                            self.logger.info(
                                f"Function {func.__name__} succeeded after {attempt} retries"
                            )
                        return result
                        
                    except Exception as e:
                        last_exception = e
                        context.retry_count = attempt
                        context.stack_trace = traceback.format_exc()
                        
                        # Log error
                        self._log_error(context, e)
                        
                        # Clinical safety check
                        if category == ErrorCategory.CLINICAL_SAFETY:
                            self._handle_clinical_safety_error(context, e)
                            break
                        
                        # If final attempt, try recovery
                        if attempt == retries:
                            if enable_recovery:
                                recovery_result = self._attempt_recovery(context, e)
                                if recovery_result is not None:
                                    return recovery_result
                            break
                        
                        # Wait before retry with exponential backoff
                        delay = min(
                            self.base_delay * (2 ** attempt),
                            self.max_delay
                        )
                        time.sleep(delay)
                
                # All attempts failed
                self._handle_final_failure(context, last_exception)
                
                # Return fallback value or re-raise
                if fallback_value is not None:
                    self.logger.warning(
                        f"Returning fallback value for {func.__name__}: {fallback_value}"
                    )
                    return fallback_value
                
                raise last_exception
            
            return wrapper
        return decorator

    def _check_memory_usage(self) -> None:
        """Check memory usage before heavy operations."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                import gc
                gc.collect()
                self.logger.warning(f"High memory usage: {memory.percent}%")
        except ImportError:
            pass

    def _log_error(self, context: ErrorContext, exception: Exception) -> None:
        """Log error with full context."""
        if not self.log_all_errors:
            return
        
        error_record = {
            'timestamp': context.timestamp,
            'context': context.to_dict(),
            'exception': {
                'type': type(exception).__name__,
                'message': str(exception),
                'args': exception.args
            }
        }
        
        # Write to error log file
        try:
            with open(self.error_log_path, 'a') as f:
                f.write(json.dumps(error_record) + '\n')
        except Exception:
            pass  # Don't fail on logging errors
        
        # Update statistics
        self.error_stats['total_errors'] += 1
        self.error_stats['by_category'][context.category.value] = (
            self.error_stats['by_category'].get(context.category.value, 0) + 1
        )
        self.error_stats['by_severity'][context.severity.value] = (
            self.error_stats['by_severity'].get(context.severity.value, 0) + 1
        )
        
        # Log to standard logger
        self.logger.error(
            f"Error in {context.function_name}: {exception} "
            f"(Category: {context.category.value}, Severity: {context.severity.value})"
        )

    def _attempt_recovery(self, context: ErrorContext, exception: Exception) -> Any:
        """Attempt automatic recovery based on error category."""
        recovery_func = self.recovery_strategies.get(context.category)
        if recovery_func:
            try:
                result = recovery_func(context, exception)
                self.logger.info(f"Recovery successful for {context.function_name}")
                return result
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
        return None

    def _handle_memory_overflow(self, context: ErrorContext, exception: Exception) -> Any:
        """Handle memory overflow errors."""
        import gc
        gc.collect()
        
        # Try to free up memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.logger.info("Attempted memory cleanup for overflow recovery")
        return None

    def _handle_model_failure(self, context: ErrorContext, exception: Exception) -> Any:
        """Handle model inference failures."""
        # Could implement model fallback logic here
        self.logger.warning(f"Model inference failed: {exception}")
        
        # Return simplified prediction or confidence score
        if 'classification' in context.function_name.lower():
            return {'prediction': 'UNKNOWN', 'confidence': 0.0, 'error': True}
        
        return None

    def _handle_data_processing_error(self, context: ErrorContext, exception: Exception) -> Any:
        """Handle data processing errors."""
        # Could implement data preprocessing fallbacks
        self.logger.warning(f"Data processing failed: {exception}")
        return None

    def _handle_network_failure(self, context: ErrorContext, exception: Exception) -> Any:
        """Handle network-related failures."""
        self.logger.warning(f"Network operation failed: {exception}")
        return None

    def _handle_clinical_safety_error(self, context: ErrorContext, exception: Exception) -> None:
        """Handle clinical safety-critical errors."""
        # Critical error - must not continue
        self.logger.critical(
            f"CLINICAL SAFETY ERROR in {context.function_name}: {exception}"
        )
        
        # Could trigger alerts, notifications, etc.
        self._trigger_safety_alert(context, exception)

    def _trigger_safety_alert(self, context: ErrorContext, exception: Exception) -> None:
        """Trigger clinical safety alerts."""
        alert_message = (
            f"CRITICAL CLINICAL ERROR: {exception} in {context.function_name}. "
            f"Patient ID: {context.patient_id}, Slide ID: {context.slide_id}"
        )
        
        # Log critical alert
        self.logger.critical(alert_message)
        
        # Could integrate with hospital alert systems here
        # For now, just ensure error is prominently logged

    def _handle_final_failure(self, context: ErrorContext, exception: Exception) -> None:
        """Handle final failure after all retries exhausted."""
        self.logger.error(
            f"Final failure in {context.function_name} after {context.retry_count} retries: {exception}"
        )

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return self.error_stats.copy()

    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'by_category': {},
            'by_severity': {},
            'recovery_success_rate': 0.0
        }


# Global error handler instance
global_error_handler = RobustErrorHandler()

# Convenience decorators
def robust_clinical(func: F) -> F:
    """Decorator for clinical safety-critical functions."""
    return global_error_handler.robust_execution(
        severity=ErrorSeverity.CRITICAL,
        category=ErrorCategory.CLINICAL_SAFETY,
        max_retries=1,
        enable_recovery=False
    )(func)

def robust_inference(func: F) -> F:
    """Decorator for model inference functions."""
    return global_error_handler.robust_execution(
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.MODEL_INFERENCE,
        max_retries=3,
        enable_recovery=True
    )(func)

def robust_data_processing(func: F) -> F:
    """Decorator for data processing functions."""
    return global_error_handler.robust_execution(
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.DATA_PROCESSING,
        max_retries=2,
        enable_recovery=True
    )(func)