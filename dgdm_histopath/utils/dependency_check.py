"""
Dependency validation and environment checking utilities.

This module provides comprehensive dependency checking without requiring
the dependencies to be installed, enabling graceful degradation.
"""

import sys
import importlib
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class DependencyChecker:
    """Advanced dependency checker with environment validation."""
    
    CORE_DEPENDENCIES = {
        "torch": ">=2.0.0",
        "torchvision": ">=0.15.0",
        "torch_geometric": ">=2.3.0",
        "numpy": ">=1.21.0",
        "scipy": ">=1.7.0",
        "scikit-learn": ">=1.0.0",
        "pandas": ">=1.3.0",
        "pillow": ">=8.3.0",
        "opencv-python": ">=4.5.0",
        "matplotlib": ">=3.4.0",
        "lightning": ">=2.0.0"
    }
    
    OPTIONAL_DEPENDENCIES = {
        "openslide-python": ">=1.1.2",
        "wandb": ">=0.15.0",
        "tensorboard": ">=2.8.0",
        "plotly": ">=5.0.0",
        "bokeh": ">=2.4.0"
    }
    
    CLINICAL_DEPENDENCIES = {
        "pydicom": ">=2.3.0",
        "SimpleITK": ">=2.2.0",
        "cryptography": ">=40.0.0"
    }

    def __init__(self):
        self.results = {}
        self.warnings = []
        self.errors = []
    
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        try:
            if sys.version_info < (3, 9):
                self.errors.append(
                    f"Python 3.9+ required, found {sys.version_info.major}.{sys.version_info.minor}"
                )
                return False
            return True
        except Exception as e:
            self.errors.append(f"Failed to check Python version: {e}")
            return False
    
    def check_dependency(self, name: str, version_req: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Check if a dependency is available and meets version requirements."""
        try:
            # Special handling for module name mapping
            import_name = self._get_import_name(name)
            
            module = importlib.import_module(import_name)
            
            # Get version
            version = self._get_module_version(module, name)
            
            if version_req and version:
                # Simple version comparison (for more complex, use packaging)
                if not self._version_meets_requirement(version, version_req):
                    self.warnings.append(
                        f"{name}: version {version} may not meet requirement {version_req}"
                    )
                    return True, version  # Still available, just version warning
            
            return True, version
            
        except ImportError:
            return False, None
        except Exception as e:
            self.warnings.append(f"Error checking {name}: {e}")
            return False, None
    
    def _get_import_name(self, package_name: str) -> str:
        """Map package names to import names."""
        mapping = {
            "opencv-python": "cv2",
            "pillow": "PIL",
            "scikit-learn": "sklearn",
            "torch-geometric": "torch_geometric",
            "pytorch-lightning": "lightning",
            "openslide-python": "openslide",
            "SimpleITK": "SimpleITK"
        }
        return mapping.get(package_name, package_name.replace("-", "_"))
    
    def _get_module_version(self, module: Any, package_name: str) -> Optional[str]:
        """Extract version from module."""
        version_attrs = ["__version__", "version", "VERSION"]
        
        for attr in version_attrs:
            if hasattr(module, attr):
                version = getattr(module, attr)
                if isinstance(version, str):
                    return version
                elif hasattr(version, "__str__"):
                    return str(version)
        
        # Try package metadata
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except:
            pass
        
        return None
    
    def _version_meets_requirement(self, version: str, requirement: str) -> bool:
        """Simple version comparison."""
        try:
            # Remove >= prefix and compare
            req_version = requirement.replace(">=", "").strip()
            
            # Simple numeric comparison
            version_parts = [int(x) for x in version.split(".") if x.isdigit()]
            req_parts = [int(x) for x in req_version.split(".") if x.isdigit()]
            
            # Pad to same length
            max_len = max(len(version_parts), len(req_parts))
            version_parts.extend([0] * (max_len - len(version_parts)))
            req_parts.extend([0] * (max_len - len(req_parts)))
            
            return version_parts >= req_parts
            
        except:
            return True  # Assume OK if can't parse
    
    def check_all_dependencies(self) -> Dict[str, Any]:
        """Check all dependency categories."""
        
        # Check Python version first
        python_ok = self.check_python_version()
        
        # Check core dependencies
        core_results = {}
        for name, version_req in self.CORE_DEPENDENCIES.items():
            available, version = self.check_dependency(name, version_req)
            core_results[name] = {
                "available": available,
                "version": version,
                "required": version_req
            }
        
        # Check optional dependencies
        optional_results = {}
        for name, version_req in self.OPTIONAL_DEPENDENCIES.items():
            available, version = self.check_dependency(name, version_req)
            optional_results[name] = {
                "available": available,
                "version": version,
                "required": version_req
            }
        
        # Check clinical dependencies
        clinical_results = {}
        for name, version_req in self.CLINICAL_DEPENDENCIES.items():
            available, version = self.check_dependency(name, version_req)
            clinical_results[name] = {
                "available": available,
                "version": version,
                "required": version_req
            }
        
        # Calculate summary statistics
        core_available = sum(1 for dep in core_results.values() if dep["available"])
        core_total = len(core_results)
        
        optional_available = sum(1 for dep in optional_results.values() if dep["available"])
        optional_total = len(optional_results)
        
        clinical_available = sum(1 for dep in clinical_results.values() if dep["available"])
        clinical_total = len(clinical_results)
        
        return {
            "python_version_ok": python_ok,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "core_dependencies": core_results,
            "optional_dependencies": optional_results,
            "clinical_dependencies": clinical_results,
            "summary": {
                "core_available": f"{core_available}/{core_total}",
                "optional_available": f"{optional_available}/{optional_total}",
                "clinical_available": f"{clinical_available}/{clinical_total}",
                "core_percentage": round(100 * core_available / core_total, 1),
                "overall_ready": python_ok and core_available >= core_total * 0.8
            },
            "warnings": self.warnings,
            "errors": self.errors
        }
    
    def check_gpu_availability(self) -> Dict[str, Any]:
        """Check GPU availability and configuration."""
        gpu_info = {
            "cuda_available": False,
            "cuda_version": None,
            "gpu_count": 0,
            "gpu_devices": [],
            "torch_cuda_available": False
        }
        
        try:
            # Check if torch is available
            torch_available, _ = self.check_dependency("torch")
            if torch_available:
                import torch
                gpu_info["torch_cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    gpu_info["cuda_available"] = True
                    gpu_info["cuda_version"] = torch.version.cuda
                    gpu_info["gpu_count"] = torch.cuda.device_count()
                    
                    for i in range(torch.cuda.device_count()):
                        gpu_info["gpu_devices"].append({
                            "id": i,
                            "name": torch.cuda.get_device_name(i),
                            "memory_total": torch.cuda.get_device_properties(i).total_memory,
                            "memory_reserved": torch.cuda.memory_reserved(i),
                            "memory_allocated": torch.cuda.memory_allocated(i)
                        })
        except Exception as e:
            gpu_info["error"] = str(e)
        
        return gpu_info
    
    def generate_installation_script(self) -> str:
        """Generate installation script for missing dependencies."""
        results = self.check_all_dependencies()
        
        missing_core = []
        missing_optional = []
        missing_clinical = []
        
        for name, info in results["core_dependencies"].items():
            if not info["available"]:
                missing_core.append(f"{name}{info['required']}")
        
        for name, info in results["optional_dependencies"].items():
            if not info["available"]:
                missing_optional.append(f"{name}{info['required']}")
        
        for name, info in results["clinical_dependencies"].items():
            if not info["available"]:
                missing_clinical.append(f"{name}{info['required']}")
        
        script_parts = ["#!/bin/bash", "# Auto-generated installation script", ""]
        
        if missing_core:
            script_parts.extend([
                "# Install core dependencies",
                f"pip install {' '.join(missing_core)}",
                ""
            ])
        
        if missing_optional:
            script_parts.extend([
                "# Install optional dependencies (recommended)",
                f"pip install {' '.join(missing_optional)}",
                ""
            ])
        
        if missing_clinical:
            script_parts.extend([
                "# Install clinical dependencies (for production use)",
                f"pip install {' '.join(missing_clinical)}",
                ""
            ])
        
        script_parts.extend([
            "# Verify installation",
            "python -c \"import dgdm_histopath; print('Installation verified!')\""
        ])
        
        return "\n".join(script_parts)
    
    def save_report(self, output_path: Path) -> None:
        """Save comprehensive dependency report."""
        results = self.check_all_dependencies()
        gpu_info = self.check_gpu_availability()
        installation_script = self.generate_installation_script()
        
        report = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "system_info": {
                "platform": sys.platform,
                "python_executable": sys.executable,
                "python_path": sys.path[:3]  # First few entries
            },
            "dependency_check": results,
            "gpu_info": gpu_info,
            "installation_script": installation_script
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Dependency report saved to {output_path}")


def quick_check() -> bool:
    """Quick dependency check for basic functionality."""
    checker = DependencyChecker()
    
    # Check minimum requirements
    essential_deps = ["numpy", "torch", "torchvision"]
    
    available_count = 0
    for dep in essential_deps:
        available, _ = checker.check_dependency(dep)
        if available:
            available_count += 1
    
    return available_count >= len(essential_deps) * 0.5  # At least 50% available


def print_dependency_summary():
    """Print a user-friendly dependency summary."""
    checker = DependencyChecker()
    results = checker.check_all_dependencies()
    
    print("=" * 60)
    print("DGDM Histopath Lab - Dependency Check")
    print("=" * 60)
    
    print(f"Python Version: {results['python_version']} " + 
          ("✓" if results['python_version_ok'] else "✗"))
    
    print(f"\nCore Dependencies: {results['summary']['core_available']} " +
          f"({results['summary']['core_percentage']}%)")
    
    print(f"Optional Dependencies: {results['summary']['optional_available']}")
    print(f"Clinical Dependencies: {results['summary']['clinical_available']}")
    
    if results['summary']['overall_ready']:
        print("\n✓ System ready for DGDM Histopath Lab!")
    else:
        print("\n⚠ Some dependencies missing. See installation guide.")
    
    if results['warnings']:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")
    
    if results['errors']:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_dependency_summary()