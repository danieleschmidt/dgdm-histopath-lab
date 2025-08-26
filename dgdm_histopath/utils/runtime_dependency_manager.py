"""
Runtime Dependency Manager for DGDM Histopath Lab
Handles optional dependencies and provides graceful degradation
"""

import importlib
import sys
import warnings
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    import_name: str
    required: bool = False
    install_command: str = ""
    fallback_available: bool = False
    description: str = ""


class DependencyManager:
    """Manages runtime dependencies and provides graceful fallbacks."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_dependencies: Dict[str, bool] = {}
        self.failed_dependencies: Set[str] = set()
        self.dependency_info: Dict[str, DependencyInfo] = {}
        
        # Register common dependencies
        self._register_dependencies()
        
        # Check all dependencies at startup
        self._check_all_dependencies()
    
    def _register_dependencies(self):
        """Register known dependencies with their information."""
        dependencies = [
            DependencyInfo(
                name="torch",
                import_name="torch",
                required=False,
                install_command="pip install torch torchvision",
                fallback_available=True,
                description="PyTorch for neural networks"
            ),
            DependencyInfo(
                name="numpy",
                import_name="numpy",
                required=False,
                install_command="pip install numpy",
                fallback_available=True,
                description="Numerical computing"
            ),
            DependencyInfo(
                name="psutil",
                import_name="psutil",
                required=False,
                install_command="pip install psutil",
                fallback_available=True,
                description="System monitoring"
            ),
            DependencyInfo(
                name="cv2",
                import_name="cv2",
                required=False,
                install_command="pip install opencv-python",
                fallback_available=True,
                description="Computer vision"
            ),
            DependencyInfo(
                name="sklearn",
                import_name="sklearn",
                required=False,
                install_command="pip install scikit-learn",
                fallback_available=True,
                description="Machine learning"
            ),
            DependencyInfo(
                name="yaml",
                import_name="yaml",
                required=False,
                install_command="pip install pyyaml",
                fallback_available=True,
                description="YAML parsing"
            ),
        ]
        
        for dep in dependencies:
            self.dependency_info[dep.name] = dep
    
    def _check_all_dependencies(self):
        """Check availability of all registered dependencies."""
        for name, info in self.dependency_info.items():
            self.available_dependencies[name] = self._check_dependency(info.import_name)
    
    def _check_dependency(self, import_name: str) -> bool:
        """Check if a dependency is available."""
        try:
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
    
    def is_available(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        return self.available_dependencies.get(dependency, False)
    
    def require(self, dependency: str) -> Any:
        """Import and return a dependency, with graceful error handling."""
        if dependency not in self.dependency_info:
            raise ValueError(f"Unknown dependency: {dependency}")
        
        info = self.dependency_info[dependency]
        
        if self.is_available(dependency):
            try:
                return importlib.import_module(info.import_name)
            except ImportError as e:
                self.failed_dependencies.add(dependency)
                if info.required:
                    raise ImportError(f"Required dependency '{dependency}' not available: {e}")
                else:
                    warnings.warn(f"Optional dependency '{dependency}' not available: {e}")
                    return None
        else:
            self.failed_dependencies.add(dependency)
            if info.required:
                raise ImportError(f"Required dependency '{dependency}' not installed. Install with: {info.install_command}")
            else:
                warnings.warn(f"Optional dependency '{dependency}' not available. Install with: {info.install_command}")
                return None
    
    def optional_import(self, dependency: str, fallback=None):
        """Import dependency with fallback value."""
        module = self.require(dependency)
        return module if module is not None else fallback
    
    def requires_dependency(self, dependency: str, fallback_return=None):
        """Decorator that checks for dependency availability."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_available(dependency):
                    info = self.dependency_info.get(dependency, DependencyInfo(name=dependency, import_name=dependency))
                    warning_msg = f"Function '{func.__name__}' requires '{dependency}' but it's not available. Install with: {info.install_command}"
                    warnings.warn(warning_msg)
                    
                    if fallback_return is not None:
                        return fallback_return
                    else:
                        raise ImportError(warning_msg)
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def fallback_implementation(self, dependency: str):
        """Decorator that provides fallback implementation when dependency is missing."""
        def decorator(fallback_func: Callable):
            def create_wrapper(original_func: Callable):
                @wraps(original_func)
                def wrapper(*args, **kwargs):
                    if self.is_available(dependency):
                        return original_func(*args, **kwargs)
                    else:
                        warnings.warn(f"Using fallback implementation for '{original_func.__name__}' due to missing '{dependency}'")
                        return fallback_func(*args, **kwargs)
                return wrapper
            return create_wrapper
        return decorator
    
    @contextmanager
    def optional_context(self, dependency: str):
        """Context manager for optional dependency usage."""
        if self.is_available(dependency):
            yield self.require(dependency)
        else:
            yield None
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get dependency status report."""
        available_count = sum(1 for available in self.available_dependencies.values() if available)
        total_count = len(self.dependency_info)
        
        status = {
            'total_dependencies': total_count,
            'available_dependencies': available_count,
            'missing_dependencies': total_count - available_count,
            'availability_percentage': (available_count / total_count * 100) if total_count > 0 else 0,
            'dependencies': {}
        }
        
        for name, info in self.dependency_info.items():
            status['dependencies'][name] = {
                'available': self.is_available(name),
                'required': info.required,
                'description': info.description,
                'install_command': info.install_command,
                'fallback_available': info.fallback_available
            }
        
        return status
    
    def install_missing_dependencies(self, optional_only: bool = True) -> List[str]:
        """Generate installation commands for missing dependencies."""
        commands = []
        
        for name, info in self.dependency_info.items():
            if not self.is_available(name):
                if not optional_only or not info.required:
                    if info.install_command:
                        commands.append(info.install_command)
        
        return commands


# Global dependency manager
global_dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance."""
    return global_dependency_manager


# Convenience functions
def is_available(dependency: str) -> bool:
    """Check if dependency is available."""
    return global_dependency_manager.is_available(dependency)


def require(dependency: str) -> Any:
    """Require a dependency with error handling."""
    return global_dependency_manager.require(dependency)


def optional_import(dependency: str, fallback=None):
    """Import with fallback."""
    return global_dependency_manager.optional_import(dependency, fallback)


def requires_dependency(dependency: str, fallback_return=None):
    """Decorator for dependency requirements."""
    return global_dependency_manager.requires_dependency(dependency, fallback_return)


def fallback_implementation(dependency: str):
    """Decorator for fallback implementations."""
    return global_dependency_manager.fallback_implementation(dependency)


# Convenience decorators for common dependencies
requires_torch = lambda fallback=None: requires_dependency("torch", fallback)
requires_numpy = lambda fallback=None: requires_dependency("numpy", fallback)
requires_psutil = lambda fallback=None: requires_dependency("psutil", fallback)
requires_opencv = lambda fallback=None: requires_dependency("cv2", fallback)
requires_sklearn = lambda fallback=None: requires_dependency("sklearn", fallback)


if __name__ == "__main__":
    # Test dependency management
    dm = get_dependency_manager()
    
    # Check status
    status = dm.get_status_report()
    print("Dependency Status Report:")
    for name, info in status['dependencies'].items():
        status_icon = "✅" if info['available'] else "❌"
        print(f"  {status_icon} {name}: {info['description']}")
        if not info['available'] and info['install_command']:
            print(f"    Install: {info['install_command']}")
    
    print(f"\nOverall: {status['available_dependencies']}/{status['total_dependencies']} dependencies available ({status['availability_percentage']:.1f}%)")
    
    # Test decorators
    @requires_torch(fallback_return="torch not available")
    def torch_function():
        torch = require("torch")
        return f"PyTorch version: {torch.__version__}"
    
    @requires_numpy(fallback_return=42)
    def numpy_function():
        np = require("numpy")
        return np.array([1, 2, 3])
    
    print(f"\nTorch function: {torch_function()}")
    print(f"NumPy function: {numpy_function()}")