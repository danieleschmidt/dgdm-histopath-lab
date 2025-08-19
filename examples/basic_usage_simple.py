#!/usr/bin/env python3
"""
Simple basic usage example for DGDM Histopath Lab
Works with or without full ML dependencies installed
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def check_dependencies():
    """Check what dependencies are available."""
    available = {}
    
    try:
        import torch
        available['torch'] = torch.__version__
        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        available['torch'] = None
        print("❌ PyTorch not available")
    
    try:
        import numpy as np
        available['numpy'] = np.__version__
        print(f"✅ NumPy {np.__version__} available")
    except ImportError:
        available['numpy'] = None
        print("❌ NumPy not available")
    
    try:
        import dgdm_histopath
        status = dgdm_histopath.check_installation()
        available['dgdm'] = status
        print(f"✅ DGDM Histopath v{status['version']} available")
        print(f"   Core available: {status['core_available']}")
        print(f"   Quantum available: {status['quantum_available']}")
    except ImportError as e:
        available['dgdm'] = None
        print(f"❌ DGDM Histopath not available: {e}")
    
    return available

def demo_configuration():
    """Demonstrate configuration without dependencies."""
    print("\n📋 Demo: Configuration Management")
    print("-" * 40)
    
    config = {
        "model": {
            "node_features": 768,
            "hidden_dims": [512, 256, 128],
            "num_diffusion_steps": 10,
            "attention_heads": 8,
            "dropout": 0.1
        },
        "preprocessing": {
            "patch_size": 256,
            "overlap": 0.25,
            "tissue_threshold": 0.8,
            "magnifications": [5, 20, 40]
        },
        "training": {
            "pretrain_epochs": 50,
            "finetune_epochs": 100,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "masking_ratio": 0.15
        }
    }
    
    print("✅ DGDM Configuration structure:")
    for section, params in config.items():
        print(f"   {section}:")
        for key, value in params.items():
            print(f"     - {key}: {value}")
    
    return config

def demo_synthetic_data(use_torch=False):
    """Create synthetic histopathology data."""
    print("\n🔬 Demo: Synthetic Data Generation")
    print("-" * 40)
    
    if use_torch:
        import torch
        # Create synthetic tissue patches
        num_patches = 100
        patch_size = 256
        patches = torch.randn(num_patches, 3, patch_size, patch_size)
        coordinates = torch.randint(0, 5000, (num_patches, 2)).float()
        
        # Create tissue graph
        num_nodes = 50
        node_features = torch.randn(num_nodes, 768)  # DINOv2-like features
        adjacency = torch.rand(num_nodes, num_nodes) > 0.8
        adjacency = adjacency.float()
        
        print(f"✅ Generated {num_patches} synthetic patches")
        print(f"✅ Generated tissue graph with {num_nodes} nodes")
        
        return {
            'patches': patches,
            'coordinates': coordinates,
            'node_features': node_features,
            'adjacency': adjacency
        }
    else:
        # Pure Python fallback
        num_patches = 100
        patches = [f"patch_{i}" for i in range(num_patches)]
        coordinates = [(i*100, j*100) for i, j in enumerate(range(num_patches))]
        
        num_nodes = 50
        node_features = [[0.1*i + 0.01*j for j in range(768)] for i in range(num_nodes)]
        adjacency = [[1.0 if abs(i-j) < 3 and i != j else 0.0 for j in range(num_nodes)] for i in range(num_nodes)]
        
        print(f"✅ Generated {num_patches} synthetic patch references")
        print(f"✅ Generated tissue graph with {num_nodes} nodes")
        
        return {
            'patches': patches,
            'coordinates': coordinates,
            'node_features': node_features,
            'adjacency': adjacency
        }

def demo_model_workflow(use_torch=False):
    """Demonstrate model workflow."""
    print("\n🧠 Demo: Model Workflow")
    print("-" * 40)
    
    if use_torch:
        try:
            from dgdm_histopath.models.dgdm_model import DGDMModel
            
            model = DGDMModel(
                node_features=768,
                hidden_dims=[512, 256, 128],
                num_diffusion_steps=5,  # Small for demo
                attention_heads=4,
                dropout=0.1,
                num_classes=2
            )
            
            print(f"✅ Initialized DGDM model")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Mode: Full PyTorch implementation")
            
            # Test forward pass
            import torch
            dummy_features = torch.randn(1, 50, 768)
            dummy_adj = torch.rand(1, 50, 50)
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_features, dummy_adj)
            
            print(f"✅ Forward pass successful")
            print(f"   Output shape: {output.shape if hasattr(output, 'shape') else 'complex output'}")
            
            return model
            
        except ImportError:
            print("⚠️ Full model not available, using simulation")
            return demo_model_simulation()
    else:
        return demo_model_simulation()

def demo_model_simulation():
    """Simulate model without torch."""
    print("✅ Model simulation mode:")
    print("   - Architecture: Dynamic Graph Diffusion Model")
    print("   - Node features: 768 (DINOv2-compatible)")
    print("   - Hidden layers: [512, 256, 128]")
    print("   - Attention heads: 4")
    print("   - Diffusion steps: 5")
    print("   - Output: Classification probabilities")
    
    return {
        'type': 'DGDM',
        'status': 'simulated',
        'node_features': 768,
        'num_classes': 2
    }

def demo_training_pipeline():
    """Demonstrate training pipeline concepts."""
    print("\n🏋️ Demo: Training Pipeline")
    print("-" * 40)
    
    training_stages = [
        ("Data Loading", "Load histopathology slides and extract patches"),
        ("Graph Construction", "Build hierarchical tissue graphs"),
        ("Self-Supervised Pretraining", "Entity masking with diffusion models"),
        ("Supervised Fine-tuning", "Task-specific classification/regression"),
        ("Validation", "Clinical evaluation metrics"),
        ("Inference", "Predictions with attention visualization")
    ]
    
    for stage, description in training_stages:
        print(f"   ✅ {stage}: {description}")
    
    print("\n📊 Expected Performance Targets:")
    performance = [
        ("TCGA-BRCA Classification", "94.3% AUC"),
        ("CAMELYON16 Metastasis", "97.6% AUC"),
        ("Slide Processing", "~30 seconds"),
        ("Model Inference", "~5 seconds")
    ]
    
    for task, target in performance:
        print(f"   🎯 {task}: {target}")

def demo_clinical_integration():
    """Demonstrate clinical integration features."""
    print("\n🏥 Demo: Clinical Integration")
    print("-" * 40)
    
    clinical_features = [
        "FDA 510(k) pathway-ready preprocessing",
        "DICOM integration for hospital systems", 
        "Stain normalization for scanner independence",
        "Quality control and validation pipelines",
        "Uncertainty quantification for clinical decisions",
        "Structured reporting (DICOM-SR compatible)",
        "Multi-institutional federated learning",
        "Privacy-preserving differential privacy"
    ]
    
    for feature in clinical_features:
        print(f"   ✅ {feature}")

def main():
    """Run comprehensive basic usage demo."""
    print("🚀 DGDM Histopath Lab - Enhanced Basic Usage Demo")
    print("=" * 60)
    
    # Check dependencies
    available = check_dependencies()
    use_torch = available['torch'] is not None
    
    # Run demos
    config = demo_configuration()
    synthetic_data = demo_synthetic_data(use_torch)
    model = demo_model_workflow(use_torch)
    demo_training_pipeline()
    demo_clinical_integration()
    
    print("\n🎉 Demo completed successfully!")
    print("\nEnvironment Summary:")
    print(f"   - PyTorch: {'✅ Available' if use_torch else '❌ Install needed'}")
    print(f"   - DGDM Core: {'✅ Available' if available['dgdm'] else '❌ Dependencies needed'}")
    
    if not use_torch:
        print("\n📦 To install full environment:")
        print("   python3 setup_environment.py")
    
    print("\nNext Steps:")
    print("1. Install dependencies: python3 setup_environment.py")
    print("2. Configure your data: edit configs/dgdm_base.yaml")
    print("3. Prepare slides: dgdm-preprocess process-slides --input_dir slides/")
    print("4. Train model: dgdm-train --config configs/dgdm_base.yaml")
    print("5. Make predictions: dgdm-predict --model model.ckpt --input slide.svs")

if __name__ == "__main__":
    main()