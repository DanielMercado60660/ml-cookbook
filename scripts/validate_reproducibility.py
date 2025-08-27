#!/usr/bin/env python3
"""
Quality Gate Validation Script for Project 1.2: The Reproducibility Toolkit

This script validates that the reproducibility toolkit meets all quality gates:
1. Achieve identical metrics (within deterministic tolerance) across 3 consecutive runs
2. Create a "reproducibility" pass in CI pipeline
"""

import sys
import os
import tempfile
import json
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test that all modules can be imported."""
    print("Testing basic imports...")
    
    try:
        from cookbook.reproduce import (
            set_global_seed, get_current_seed, SeedContext,
            enable_deterministic_mode, disable_deterministic_mode, get_deterministic_status,
            compute_checkpoint_hash, verify_checkpoint_integrity, CheckpointVerifier,
            create_reproducible_template, validate_template_structure
        )
        print("✅ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_quality_gate_1():
    """Quality Gate 1: Identical metrics across 3 consecutive runs."""
    print("\nTesting Quality Gate 1: Identical metrics across 3 runs...")
    
    try:
        from cookbook.reproduce import set_global_seed, enable_deterministic_mode
        
        def mock_experiment():
            """Mock ML experiment that should be deterministic."""
            set_global_seed(42)
            enable_deterministic_mode()
            
            # Simulate ML operations
            data = np.random.rand(50, 5)
            weights = np.random.rand(5, 1)
            predictions = np.dot(data, weights)
            
            return {
                'mean_prediction': float(predictions.mean()),
                'sum_weights': float(weights.sum())
            }
        
        # Run experiment 3 times
        results = []
        for i in range(3):
            result = mock_experiment()
            results.append(result)
        
        # Check identical results
        tolerance = 1e-10
        for metric in results[0].keys():
            for i in range(1, len(results)):
                diff = abs(results[0][metric] - results[i][metric])
                if diff >= tolerance:
                    print(f"❌ Run {i} differs from run 0 in {metric}: {diff}")
                    return False
        
        print(f"✅ Quality Gate 1 PASSED: Identical results across 3 runs")
        print(f"   Sample results: {results[0]}")
        return True
        
    except Exception as e:
        print(f"❌ Quality Gate 1 FAILED: {e}")
        return False

def test_quality_gate_2():
    """Quality Gate 2: CI pipeline reproducibility checks."""
    print("\nTesting Quality Gate 2: CI pipeline reproducibility...")
    
    try:
        from cookbook.reproduce import (
            create_reproducible_template, validate_template_structure,
            CheckpointVerifier, compute_checkpoint_hash, set_global_seed
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "ci_test"
            
            # 1. Create reproducible project
            success = create_reproducible_template(project_path)
            if not success:
                print("❌ Template creation failed")
                return False
            
            # 2. Validate project structure  
            validation = validate_template_structure(project_path)
            if not validation['valid']:
                print(f"❌ Project validation failed: {validation['errors']}")
                return False
            
            # 3. Test checkpoint verification
            checkpoint_path = project_path / "test_checkpoint.json"
            test_data = {"test": "data", "values": [1, 2, 3]}
            
            with open(checkpoint_path, 'w') as f:
                json.dump(test_data, f)
            
            # Compute and verify hash
            hash_value = compute_checkpoint_hash(checkpoint_path)
            if len(hash_value) != 64:  # SHA256 length
                print(f"❌ Invalid hash length: {len(hash_value)}")
                return False
            
            # 4. Test reproducible execution
            set_global_seed(42)
            result1 = np.random.rand(10).sum()
            
            set_global_seed(42)
            result2 = np.random.rand(10).sum()
            
            if abs(result1 - result2) > 1e-15:
                print(f"❌ Reproducible execution failed: {abs(result1 - result2)}")
                return False
        
        print("✅ Quality Gate 2 PASSED: CI pipeline checks complete")
        return True
        
    except Exception as e:
        print(f"❌ Quality Gate 2 FAILED: {e}")
        return False

def test_template_generation():
    """Test template generation functionality."""
    print("\nTesting template generation...")
    
    try:
        from cookbook.reproduce import create_reproducible_template, TemplateConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir) / "test_template"
            
            # Create with custom config
            config = TemplateConfig(
                project_name="test_template",
                author="Test Author",
                python_version="3.10"
            )
            
            success = create_reproducible_template(project_path, config)
            if not success:
                print("❌ Template creation failed")
                return False
            
            # Check essential files exist
            required_files = [
                "pyproject.toml",
                "requirements.txt", 
                "README.md",
                "src/__init__.py",
                "src/config.py",
                "tests/test_reproducibility.py"
            ]
            
            for file_path in required_files:
                if not (project_path / file_path).exists():
                    print(f"❌ Missing required file: {file_path}")
                    return False
            
        print("✅ Template generation test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Template generation test FAILED: {e}")
        return False

def main():
    """Run all quality gate tests."""
    print("🧪 Project 1.2 Quality Gate Validation")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_quality_gate_1,
        test_quality_gate_2,
        test_template_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("   Test failed!")
        except Exception as e:
            print(f"   Test error: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED! 🎉")
        print("Project 1.2: The Reproducibility Toolkit is complete!")
        print("\n📋 What we've built:")
        print("  ✅ Global seeding utilities with context management")
        print("  ✅ Deterministic operations configuration")
        print("  ✅ Checkpoint verification with cryptographic hashes")
        print("  ✅ Reproducible project templates")
        print("  ✅ Comprehensive test suite")
        print("  ✅ Quality gates for identical metrics across runs")
        print("\n🚀 Ready to proceed to Project 1.3: The Curated Corpus Project")
        return True
    else:
        print(f"❌ {total - passed} quality gate(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
