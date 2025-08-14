#!/usr/bin/env python3
"""Validate progressive quality gates implementation."""

import sys
import importlib
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    modules_to_test = [
        'dgdm_histopath.testing.progressive_quality_gates',
        'dgdm_histopath.testing.robust_quality_runner', 
        'dgdm_histopath.testing.scalable_quality_gates',
        'dgdm_histopath.testing.monitoring_health_checks'
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            results[module_name] = "✅ SUCCESS"
        except Exception as e:
            results[module_name] = f"❌ FAILED: {e}"
    
    return results

def test_configuration():
    """Test configuration classes."""
    try:
        # Import with minimal dependencies
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Test enum imports first
        from dgdm_histopath.testing.progressive_quality_gates import ProjectMaturity
        
        maturity_levels = [m.value for m in ProjectMaturity]
        expected_levels = ['greenfield', 'development', 'staging', 'production']
        
        if set(maturity_levels) == set(expected_levels):
            return "✅ ProjectMaturity enum configured correctly"
        else:
            return f"❌ ProjectMaturity enum mismatch: {maturity_levels}"
            
    except Exception as e:
        return f"❌ Configuration test failed: {e}"

def test_cli_structure():
    """Test CLI structure."""
    try:
        from dgdm_histopath.cli.quality_gates import app
        if app is not None:
            return "✅ CLI app structure exists"
        else:
            return "❌ CLI app is None"
    except Exception as e:
        return f"❌ CLI test failed: {e}"

def test_file_structure():
    """Test that all required files exist."""
    required_files = [
        'dgdm_histopath/testing/progressive_quality_gates.py',
        'dgdm_histopath/testing/robust_quality_runner.py',
        'dgdm_histopath/testing/scalable_quality_gates.py',
        'dgdm_histopath/testing/monitoring_health_checks.py',
        'dgdm_histopath/cli/quality_gates.py',
        'tests/test_progressive_quality_gates.py',
        'tests/test_performance_benchmarks.py',
        '.github/workflows/progressive-quality-gates.yml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if not missing_files:
        return "✅ All required files exist"
    else:
        return f"❌ Missing files: {missing_files}"

def main():
    """Run validation tests."""
    print("DGDM Progressive Quality Gates - Validation Report")
    print("=" * 60)
    
    # Test file structure
    print("\n📁 File Structure:")
    print(test_file_structure())
    
    # Test configuration
    print("\n⚙️  Configuration:")
    print(test_configuration())
    
    # Test CLI structure
    print("\n🖥️  CLI Structure:")
    print(test_cli_structure())
    
    # Test imports
    print("\n📦 Module Imports:")
    import_results = test_imports()
    for module, result in import_results.items():
        print(f"  {module}: {result}")
    
    # Summary
    print("\n📊 Summary:")
    total_tests = 3 + len(import_results)
    
    successes = (
        1 if "✅" in test_file_structure() else 0 +
        1 if "✅" in test_configuration() else 0 +
        1 if "✅" in test_cli_structure() else 0 +
        len([r for r in import_results.values() if "✅" in r])
    )
    
    print(f"Tests passed: {successes}/{total_tests}")
    
    if successes == total_tests:
        print("🎉 All validations passed!")
        return 0
    else:
        print("⚠️  Some validations failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())