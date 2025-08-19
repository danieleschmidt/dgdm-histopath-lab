"""
Robust Environment Management for DGDM Histopath Lab
Handles dependency checking, graceful degradation, and environment validation
"""

import sys
import os
import warnings
import subprocess
import importlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

class EnvironmentValidator:
    """Validates and manages environment dependencies robustly."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.capabilities = {}
        self.fallback_mode = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup robust logging that works in any environment."""
        logger = logging.getLogger('dgdm_environment')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def check_python_version(self) -> bool:
        """Check Python version with detailed reporting."""
        version = sys.version_info
        required_major, required_minor = 3, 9
        
        if version.major < required_major or (version.major == required_major and version.minor < required_minor):
            self.logger.error(
                f"Python {required_major}.{required_minor}+ required, "
                f"found {version.major}.{version.minor}.{version.micro}"
            )
            return False
        
        self.logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} OK")
        return True
    
    def check_dependency(self, package_name: str, import_name: str = None) -> Tuple[bool, Optional[str]]:
        """Check if a dependency is available with version info."""
        if import_name is None:
            import_name = package_name.replace('-', '_')
        
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, '__version__', 'unknown')
            self.logger.debug(f"✅ {package_name} {version} available")
            return True, version
        except ImportError as e:
            self.logger.debug(f"❌ {package_name} not available: {e}")
            return False, None
    
    def validate_ml_dependencies(self) -> Dict[str, Any]:
        """Validate core ML dependencies with graceful fallbacks."""
        ml_deps = {
            'torch': 'torch',
            'numpy': 'numpy', 
            'scipy': 'scipy',
            'scikit-learn': 'sklearn',
            'pandas': 'pandas',
            'pillow': 'PIL',
            'opencv-python': 'cv2'
        }
        
        results = {}
        for package, import_name in ml_deps.items():
            available, version = self.check_dependency(package, import_name)
            results[package] = {
                'available': available,
                'version': version,
                'critical': package in ['torch', 'numpy']
            }
        
        # Check critical dependencies
        critical_missing = [
            pkg for pkg, info in results.items() 
            if info['critical'] and not info['available']
        ]
        
        if critical_missing:
            self.logger.warning(
                f"Critical dependencies missing: {critical_missing}. "
                "Enabling fallback mode."
            )
            self.fallback_mode = True
        
        return results
    
    def validate_dgdm_components(self) -> Dict[str, bool]:
        """Validate DGDM-specific components."""
        components = {
            'core_models': 'dgdm_histopath.models.dgdm_model',
            'preprocessing': 'dgdm_histopath.preprocessing.slide_processor',
            'training': 'dgdm_histopath.training.trainer',
            'evaluation': 'dgdm_histopath.evaluation.predictor',
            'quantum': 'dgdm_histopath.quantum.quantum_planner',
            'research': 'dgdm_histopath.research.experiment_framework'
        }
        
        results = {}
        for component, module_path in components.items():
            try:
                importlib.import_module(module_path)
                results[component] = True
                self.logger.debug(f"✅ {component} component available")
            except ImportError as e:
                results[component] = False
                self.logger.debug(f"⚠️ {component} component not available: {e}")
        
        return results
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources and capabilities."""
        resources = {}
        
        try:
            import psutil
            resources['memory_gb'] = psutil.virtual_memory().total / (1024**3)
            resources['cpu_count'] = psutil.cpu_count()
            resources['disk_free_gb'] = psutil.disk_usage('/').free / (1024**3)
            resources['psutil_available'] = True
        except ImportError:
            # Fallback resource estimation
            resources['memory_gb'] = 'unknown'
            resources['cpu_count'] = os.cpu_count() or 1
            resources['disk_free_gb'] = 'unknown'
            resources['psutil_available'] = False
        
        # Check GPU availability
        resources['cuda_available'] = False
        resources['gpu_count'] = 0
        
        try:
            import torch
            if torch.cuda.is_available():
                resources['cuda_available'] = True
                resources['gpu_count'] = torch.cuda.device_count()
                resources['gpu_names'] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(resources['gpu_count'])
                ]
        except ImportError:
            pass
        
        return resources
    
    def install_missing_dependencies(self, dependencies: List[str]) -> bool:
        """Attempt to install missing dependencies."""
        self.logger.info(f"Attempting to install: {dependencies}")
        
        for dep in dependencies:
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', dep],
                    capture_output=True, text=True, check=True
                )
                self.logger.info(f"✅ Successfully installed {dep}")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"❌ Failed to install {dep}: {e.stderr}")
                return False
        
        return True
    
    def generate_environment_report(self) -> Dict[str, Any]:
        """Generate comprehensive environment report."""
        report = {
            'timestamp': str(Path(__file__).stat().st_mtime),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'fallback_mode': self.fallback_mode
        }
        
        # Add dependency status
        report['ml_dependencies'] = self.validate_ml_dependencies()
        report['dgdm_components'] = self.validate_dgdm_components()
        report['system_resources'] = self.check_system_resources()
        
        # Calculate capability score
        total_deps = len(report['ml_dependencies'])
        available_deps = sum(1 for info in report['ml_dependencies'].values() if info['available'])
        report['dependency_score'] = available_deps / total_deps if total_deps > 0 else 0
        
        total_components = len(report['dgdm_components'])
        available_components = sum(report['dgdm_components'].values())
        report['component_score'] = available_components / total_components if total_components > 0 else 0
        
        return report
    
    def suggest_installation_steps(self, report: Dict[str, Any]) -> List[str]:
        """Suggest installation steps based on environment report."""
        steps = []
        
        # Check critical dependencies
        missing_critical = [
            pkg for pkg, info in report['ml_dependencies'].items()
            if info['critical'] and not info['available']
        ]
        
        if missing_critical:
            steps.append("Install critical ML dependencies:")
            if 'torch' in missing_critical:
                steps.append("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            if 'numpy' in missing_critical:
                steps.append("  pip install numpy scipy scikit-learn")
        
        # Check other dependencies
        missing_optional = [
            pkg for pkg, info in report['ml_dependencies'].items()
            if not info['critical'] and not info['available']
        ]
        
        if missing_optional:
            steps.append("Install optional dependencies:")
            steps.append(f"  pip install {' '.join(missing_optional)}")
        
        # Check system requirements
        resources = report['system_resources']
        if isinstance(resources.get('memory_gb'), (int, float)) and resources['memory_gb'] < 8:
            steps.append("⚠️ Warning: Less than 8GB RAM detected. Consider upgrading for large datasets.")
        
        if not resources['cuda_available']:
            steps.append("💡 Tip: Install CUDA-enabled PyTorch for GPU acceleration")
        
        return steps

class RobustDependencyManager:
    """Manages dependencies with automatic fallbacks and error recovery."""
    
    def __init__(self):
        self.validator = EnvironmentValidator()
        self.import_cache = {}
        self.fallback_implementations = {}
        
    def safe_import(self, module_name: str, fallback=None):
        """Safely import module with caching and fallback."""
        if module_name in self.import_cache:
            return self.import_cache[module_name]
        
        try:
            module = importlib.import_module(module_name)
            self.import_cache[module_name] = module
            return module
        except ImportError as e:
            self.validator.logger.warning(f"Failed to import {module_name}: {e}")
            if fallback is not None:
                self.import_cache[module_name] = fallback
                return fallback
            raise
    
    def get_torch_or_fallback(self):
        """Get PyTorch or provide fallback implementations."""
        try:
            return self.safe_import('torch')
        except ImportError:
            # Provide minimal fallback for tensor operations
            class TorchFallback:
                @staticmethod
                def randn(*shape):
                    import random
                    if len(shape) == 1:
                        return [random.gauss(0, 1) for _ in range(shape[0])]
                    # Nested list for multi-dimensional
                    def _randn_recursive(dims):
                        if len(dims) == 1:
                            return [random.gauss(0, 1) for _ in range(dims[0])]
                        return [_randn_recursive(dims[1:]) for _ in range(dims[0])]
                    return _randn_recursive(shape)
                
                @staticmethod
                def rand(*shape):
                    import random
                    if len(shape) == 1:
                        return [random.random() for _ in range(shape[0])]
                    def _rand_recursive(dims):
                        if len(dims) == 1:
                            return [random.random() for _ in range(dims[0])]
                        return [_rand_recursive(dims[1:]) for _ in range(dims[0])]
                    return _rand_recursive(shape)
                
                cuda = type('CudaFallback', (), {'is_available': lambda: False})()
                
            return TorchFallback()
    
    def validate_environment_or_fail_gracefully(self) -> Dict[str, Any]:
        """Validate environment and provide actionable feedback."""
        report = self.validator.generate_environment_report()
        
        if report['dependency_score'] < 0.5:
            self.validator.logger.warning(
                "Less than 50% of dependencies available. "
                "Some functionality will be limited."
            )
        
        if report['component_score'] < 0.7:
            self.validator.logger.warning(
                "DGDM components not fully available. "
                "Check installation integrity."
            )
        
        # Provide installation guidance
        steps = self.validator.suggest_installation_steps(report)
        if steps:
            self.validator.logger.info("Suggested installation steps:")
            for step in steps:
                self.validator.logger.info(f"  {step}")
        
        return report

# Global instance for easy access
robust_env = RobustDependencyManager()

def check_and_setup_environment() -> Dict[str, Any]:
    """Main entry point for environment validation."""
    return robust_env.validate_environment_or_fail_gracefully()

def get_capability_report() -> str:
    """Get human-readable capability report."""
    report = check_and_setup_environment()
    
    lines = [
        "🔍 DGDM Environment Capability Report",
        "=" * 50,
        f"Python: {report['python_version']} on {report['platform']}",
        f"Dependency Score: {report['dependency_score']:.1%}",
        f"Component Score: {report['component_score']:.1%}",
        f"Fallback Mode: {'Yes' if report['fallback_mode'] else 'No'}",
        "",
        "ML Dependencies:"
    ]
    
    for pkg, info in report['ml_dependencies'].items():
        status = "✅" if info['available'] else "❌"
        version = f" ({info['version']})" if info['version'] else ""
        critical = " [CRITICAL]" if info['critical'] else ""
        lines.append(f"  {status} {pkg}{version}{critical}")
    
    lines.extend([
        "",
        "DGDM Components:"
    ])
    
    for component, available in report['dgdm_components'].items():
        status = "✅" if available else "❌"
        lines.append(f"  {status} {component}")
    
    resources = report['system_resources']
    lines.extend([
        "",
        "System Resources:",
        f"  CPU Cores: {resources['cpu_count']}",
        f"  Memory: {resources['memory_gb']} GB" if isinstance(resources['memory_gb'], (int, float)) else "  Memory: Unknown",
        f"  GPU: {'✅' if resources['cuda_available'] else '❌'} ({resources['gpu_count']} devices)"
    ])
    
    return "\n".join(lines)

if __name__ == "__main__":
    print(get_capability_report())