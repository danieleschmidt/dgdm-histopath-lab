"""Enhanced logging utilities with monitoring and security features."""

import logging
import logging.handlers
import sys
import os
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import threading
from contextlib import contextmanager


class SecurityAuditFilter(logging.Filter):
    """Filter to detect and flag security-related events."""
    
    SECURITY_KEYWORDS = [
        'password', 'token', 'key', 'secret', 'credential',
        'unauthorized', 'access_denied', 'breach', 'attack',
        'injection', 'xss', 'csrf', 'malicious'
    ]
    
    def filter(self, record):
        # Check for sensitive information
        message = record.getMessage().lower()
        for keyword in self.SECURITY_KEYWORDS:
            if keyword in message:
                record.security_alert = True
                break
        return True


class HealthMonitor:
    """Monitor system health and log metrics."""
    
    def __init__(self):
        self.metrics = {
            'errors': 0,
            'warnings': 0,
            'memory_usage': 0,
            'processing_time': [],
            'gpu_usage': 0
        }
        self.lock = threading.Lock()
        
    def record_error(self):
        with self.lock:
            self.metrics['errors'] += 1
            
    def record_warning(self):
        with self.lock:
            self.metrics['warnings'] += 1
            
    def record_processing_time(self, duration: float):
        with self.lock:
            self.metrics['processing_time'].append(duration)
            # Keep only last 100 measurements
            if len(self.metrics['processing_time']) > 100:
                self.metrics['processing_time'] = self.metrics['processing_time'][-100:]
                
    def get_health_status(self) -> Dict[str, Any]:
        with self.lock:
            avg_time = sum(self.metrics['processing_time']) / len(self.metrics['processing_time']) if self.metrics['processing_time'] else 0
            return {
                'status': 'healthy' if self.metrics['errors'] < 10 else 'unhealthy',
                'errors': self.metrics['errors'],
                'warnings': self.metrics['warnings'],
                'avg_processing_time': avg_time,
                'timestamp': datetime.now().isoformat()
            }


# Global health monitor instance
health_monitor = HealthMonitor()


class EnhancedFormatter(logging.Formatter):
    """Enhanced formatter with structured logging support."""
    
    def format(self, record):
        # Add extra context
        record.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        record.process_id = os.getpid()
        record.thread_id = threading.get_ident()
        
        # Format timestamp in ISO format
        if self.datefmt:
            record.asctime = self.formatTime(record, self.datefmt)
        else:
            record.asctime = datetime.fromtimestamp(record.created).isoformat()
            
        # Add security flag if present
        if hasattr(record, 'security_alert'):
            record.levelname = f"SECURITY-{record.levelname}"
            
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_security_audit: bool = True,
    structured_logging: bool = False
):
    """Setup enhanced logging configuration with monitoring and security features."""
    
    try:
        if format_string is None:
            if structured_logging:
                format_string = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "hostname": "%(hostname)s", "pid": %(process_id)d, "tid": %(thread_id)d}'
            else:
                format_string = "%(asctime)s [%(levelname)s] %(name)s (%(hostname)s:%(process_id)d:%(thread_id)d) - %(message)s"
                
        # Validate log level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}. Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
            
        # Configure root logger
        logging.root.setLevel(numeric_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Create enhanced formatter
        formatter = EnhancedFormatter(format_string)
        
        # Console handler with error handling
        try:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            
            if enable_security_audit:
                console_handler.addFilter(SecurityAuditFilter())
                
            logging.root.addHandler(console_handler)
        except Exception as e:
            print(f"Warning: Failed to setup console logging: {e}", file=sys.stderr)
            
        # File handler with rotation if specified
        if log_file:
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use rotating file handler for log management
                file_handler = logging.handlers.RotatingFileHandler(
                    log_path,
                    maxBytes=max_file_size,
                    backupCount=backup_count
                )
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                
                if enable_security_audit:
                    file_handler.addFilter(SecurityAuditFilter())
                    
                logging.root.addHandler(file_handler)
                
            except Exception as e:
                print(f"Warning: Failed to setup file logging: {e}", file=sys.stderr)
                
        # Setup custom error handler
        class HealthMonitorHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    health_monitor.record_error()
                elif record.levelno >= logging.WARNING:
                    health_monitor.record_warning()
                    
        health_handler = HealthMonitorHandler()
        logging.root.addHandler(health_handler)
        
        # Log successful setup
        logger = logging.getLogger(__name__)
        logger.info(f"Logging system initialized with level {level}")
        if log_file:
            logger.info(f"Log file: {log_file} (max_size: {max_file_size}, backups: {backup_count})")
        if enable_security_audit:
            logger.info("Security audit logging enabled")
            
    except Exception as e:
        print(f"Critical error setting up logging: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str):
    """Context manager to log execution time of operations."""
    start_time = time.time()
    try:
        logger.info(f"Starting {operation}")
        yield
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Failed {operation} after {duration:.2f}s: {e}")
        health_monitor.record_processing_time(duration)
        raise
    else:
        duration = time.time() - start_time
        logger.info(f"Completed {operation} in {duration:.2f}s")
        health_monitor.record_processing_time(duration)


def log_system_info():
    """Log system information for debugging and monitoring."""
    logger = logging.getLogger(__name__)
    try:
        import platform
        import psutil
        
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python: {platform.python_version()}")
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        logger.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA devices: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        except ImportError:
            logger.debug("PyTorch not available for GPU detection")
            
    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with enhanced features."""
    logger = logging.getLogger(name)
    
    # Add custom methods
    def log_with_context(level, msg, **kwargs):
        extra = {
            'context': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        logger.log(level, f"{msg} | Context: {json.dumps(kwargs)}", extra=extra)
        
    logger.log_with_context = log_with_context
    return logger