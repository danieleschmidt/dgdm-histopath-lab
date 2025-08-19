#!/usr/bin/env python3
"""
Environment Setup Script for DGDM Histopath Lab
Automatically configures development environment with all dependencies
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def run_command(cmd, check=True, capture_output=True):
    """Run shell command with error handling."""
    try:
        result = subprocess.run(
            cmd, shell=True, check=check, 
            capture_output=capture_output, text=True
        )
        return result.stdout if capture_output else None
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e.stderr if e.stderr else str(e)}")
        return None

def check_python_version():
    """Ensure Python 3.9+ is available."""
    version = sys.version_info
    if version.major < 3 or version.minor < 9:
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")

def install_core_dependencies():
    """Install core ML dependencies in lightweight mode."""
    print("📦 Installing core dependencies...")
    
    # Essential packages for CPU-only development
    core_packages = [
        "numpy>=1.21.0",
        "scipy>=1.7.0", 
        "pandas>=1.3.0",
        "pillow>=8.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "pydantic>=2.0.0"
    ]
    
    for package in core_packages:
        print(f"Installing {package}...")
        result = run_command(f"pip3 install '{package}'")
        if result is None:
            print(f"⚠️ Failed to install {package}, continuing...")

def install_torch_cpu():
    """Install PyTorch CPU version for development."""
    print("🔥 Installing PyTorch CPU version...")
    run_command("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

def setup_development_env():
    """Setup complete development environment."""
    print("🛠️ Setting up development environment...")
    
    # Create necessary directories
    dirs = ["logs", "checkpoints", "data", "results"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Install development dependencies
    dev_packages = [
        "pytest>=7.0.0",
        "black>=22.0.0", 
        "flake8>=5.0.0"
    ]
    
    for package in dev_packages:
        run_command(f"pip3 install '{package}'")

def validate_installation():
    """Validate that installation completed successfully."""
    print("🔍 Validating installation...")
    
    try:
        import numpy
        import pandas
        import yaml
        print("✅ Core packages installed successfully")
        
        # Try importing torch
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} installed")
        except ImportError:
            print("⚠️ PyTorch not available")
            
        # Test dgdm_histopath import
        sys.path.insert(0, str(Path(__file__).parent))
        import dgdm_histopath
        status = dgdm_histopath.check_installation()
        print(f"✅ DGDM Histopath {status['version']} imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 DGDM Histopath Lab Environment Setup")
    print("=" * 50)
    
    check_python_version()
    install_core_dependencies()
    install_torch_cpu()
    setup_development_env()
    
    if validate_installation():
        print("\n🎉 Environment setup completed successfully!")
        print("\nNext steps:")
        print("1. Run: python3 examples/basic_usage.py")
        print("2. Train: python3 -m dgdm_histopath.cli.train --help")
        print("3. Test: pytest tests/ -v")
    else:
        print("\n❌ Environment setup encountered issues")
        print("Please check error messages above")

if __name__ == "__main__":
    main()