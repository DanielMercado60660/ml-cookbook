#!/usr/bin/env python3
"""
Demonstration script for Project 1.2: The Reproducibility Toolkit

This script demonstrates all the key features of the reproducibility toolkit:
- Global seed management
- Deterministic operations  
- Checkpoint verification
- Project template generation
"""

import sys
import tempfile
import numpy as np
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def demo_seeding():
    """Demonstrate global seeding capabilities."""
    print("🌱 Demo 1: Global Seeding")
    print("-" * 30)
    
    from cookbook.reproduce import set_global_seed, SeedContext, verify_deterministic_execution
    
    # Set global seed
    set_global_seed(42)
    print("Set global seed to 42")
    
    # Generate some random data
    data1 = np.random.rand(5)
    print(f"Random data (run 1): {data1[:3]}")
    
    # Reset seed and generate again
    set_global_seed(42)
    data2 = np.random.rand(5)
    print(f"Random data (run 2): {data2[:3]}")
    
    # Check if identical
    identical = np.allclose(data1, data2)
    print(f"Identical results: {identical} ✅" if identical else f"Identical results: {identical} ❌")
    
    # Demo seed context
    print("\nUsing SeedContext for temporary seed change...")
    set_global_seed(42)
    original = np.random.rand(3)
    
    with SeedContext(123):
        different = np.random.rand(3)
    
    back_to_original = np.random.rand(3)
    
    print(f"Original seed (42): {original}")
    print(f"Context seed (123): {different}")
    print(f"Back to original: {back_to_original}")
    
    # Verify deterministic execution
    def test_function():
        return np.random.rand(10).sum()
    
    set_global_seed(42)
    is_deterministic = verify_deterministic_execution(test_function)
    print(f"Function is deterministic: {is_deterministic} ✅" if is_deterministic else f"Function is deterministic: {is_deterministic} ❌")
    

def demo_deterministic_operations():
    """Demonstrate deterministic operations configuration."""
    print("\n🔒 Demo 2: Deterministic Operations")
    print("-" * 35)
    
    from cookbook.reproduce import enable_deterministic_mode, get_deterministic_status
    
    # Check status before
    status_before = get_deterministic_status()
    print("Deterministic status (before):")
    for key, value in status_before.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} settings")
        else:
            print(f"  {key}: {value}")
    
    # Enable deterministic mode
    print("\nEnabling deterministic mode...")
    previous_state = enable_deterministic_mode()
    
    # Check status after
    status_after = get_deterministic_status()
    print("\nDeterministic status (after):")
    for key, value in status_after.items():
        if isinstance(value, dict) and key == 'environment':
            for env_key, env_value in value.items():
                if env_value:
                    print(f"  {env_key}: {env_value}")


def demo_checkpoint_verification():
    """Demonstrate checkpoint verification."""
    print("\n🔐 Demo 3: Checkpoint Verification")
    print("-" * 35)
    
    from cookbook.reproduce import CheckpointVerifier, compute_checkpoint_hash
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some mock checkpoints
        checkpoints = {}
        for epoch in range(3):
            checkpoint_data = {
                "epoch": epoch,
                "model_state": np.random.rand(10).tolist(),
                "optimizer_state": {"lr": 0.001, "momentum": 0.9},
                "metrics": {"loss": np.random.rand(), "accuracy": 0.8 + 0.05 * epoch}
            }
            
            checkpoint_path = temp_path / f"checkpoint_epoch_{epoch}.json"
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Compute hash
            hash_value = compute_checkpoint_hash(checkpoint_path)
            checkpoints[f"checkpoint_epoch_{epoch}.json"] = {
                "hash": hash_value,
                "data": checkpoint_data
            }
            
            print(f"Created checkpoint_epoch_{epoch}.json")
            print(f"  Hash: {hash_value[:12]}...")
            print(f"  Accuracy: {checkpoint_data['metrics']['accuracy']:.3f}")
        
        # Initialize verifier
        verifier = CheckpointVerifier(temp_path)
        
        # Register all checkpoints
        print(f"\nRegistering checkpoints...")
        for filename, info in checkpoints.items():
            verifier.register_checkpoint(
                filename,
                metadata={
                    "epoch": info["data"]["epoch"],
                    "accuracy": info["data"]["metrics"]["accuracy"]
                }
            )
        
        # Verify all checkpoints
        print(f"\nVerifying checkpoint integrity...")
        verification_results = verifier.verify_all_checkpoints()
        
        for filename, is_valid in verification_results.items():
            status = "✅" if is_valid else "❌"
            print(f"  {filename}: {is_valid} {status}")
        
        # Export verification report
        report_path = temp_path / "verification_report.md"
        verifier.export_verification_report(report_path)
        print(f"\nVerification report saved to: {report_path.name}")


def demo_project_templates():
    """Demonstrate project template generation."""
    print("\n📁 Demo 4: Project Templates")
    print("-" * 30)
    
    from cookbook.reproduce import create_reproducible_template, TemplateConfig, validate_template_structure
    
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir) / "demo_experiment"
        
        # Create template with custom config
        config = TemplateConfig(
            project_name="demo_experiment",
            author="ML Cookbook Demo",
            python_version="3.10",
            frameworks=["pytorch"],
            include_docker=True
        )
        
        print(f"Creating project template: {project_path.name}")
        success = create_reproducible_template(project_path, config)
        
        if success:
            print("✅ Template created successfully!")
            
            # List created files
            print("\nCreated files:")
            for item in sorted(project_path.rglob("*")):
                if item.is_file():
                    relative_path = item.relative_to(project_path)
                    print(f"  {relative_path}")
            
            # Validate structure
            print(f"\nValidating project structure...")
            validation = validate_template_structure(project_path)
            
            print(f"Validation score: {validation['score']:.2f}")
            print(f"Validation status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
            
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
            
            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  - {error}")
            
            # Show sample content from key files
            print(f"\nSample content from key files:")
            
            # Show config.py content
            config_path = project_path / "src" / "config.py"
            if config_path.exists():
                config_content = config_path.read_text()
                lines = config_content.split('\n')
                print(f"\n📄 src/config.py (first 10 lines):")
                for i, line in enumerate(lines[:10]):
                    print(f"  {i+1:2d}: {line}")
            
            # Show requirements.txt
            req_path = project_path / "requirements.txt"
            if req_path.exists():
                req_content = req_path.read_text().strip()
                print(f"\n📄 requirements.txt:")
                for line in req_content.split('\n')[:8]:
                    if line.strip():
                        print(f"  {line}")
        
        else:
            print("❌ Template creation failed")


def demo_complete_workflow():
    """Demonstrate complete reproducible workflow."""
    print("\n🔄 Demo 5: Complete Reproducible Workflow")
    print("-" * 42)
    
    from cookbook.reproduce import (
        set_global_seed, enable_deterministic_mode,
        CheckpointVerifier, create_reproducible_template
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Create reproducible project
        project_path = Path(temp_dir) / "complete_demo"
        create_reproducible_template(project_path)
        print("1. ✅ Created reproducible project template")
        
        # Step 2: Set up reproducible environment
        set_global_seed(42)
        enable_deterministic_mode()
        print("2. ✅ Configured reproducible environment")
        
        # Step 3: Simulate ML experiment
        print("3. 🧪 Running simulated ML experiment...")
        
        # Generate synthetic dataset
        np.random.seed(42)  # Explicit seed for this demo
        X = np.random.randn(100, 20)
        y = np.random.randint(0, 2, 100)
        
        # Simulate training loop with checkpointing
        verifier = CheckpointVerifier(project_path / "checkpoints")
        
        model_weights = np.random.randn(20, 1) * 0.01
        learning_rate = 0.01
        
        for epoch in range(5):
            # Simulate gradient update
            predictions = 1 / (1 + np.exp(-X @ model_weights))  # Simple logistic regression
            gradient = X.T @ (predictions.flatten() - y) / len(y)
            model_weights -= learning_rate * gradient.reshape(-1, 1)
            
            # Compute loss
            loss = -np.mean(y * np.log(predictions.flatten() + 1e-8) + 
                           (1 - y) * np.log(1 - predictions.flatten() + 1e-8))
            
            # Save checkpoint every 2 epochs
            if epoch % 2 == 0 or epoch == 4:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_weights": model_weights.tolist(),
                    "loss": float(loss),
                    "learning_rate": learning_rate
                }
                
                checkpoint_path = project_path / "checkpoints" / f"model_epoch_{epoch}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f)
                
                # Register with verifier
                verifier.register_checkpoint(
                    f"model_epoch_{epoch}.json",
                    metadata={"epoch": epoch, "loss": float(loss)}
                )
                
                print(f"   Epoch {epoch}: loss={loss:.4f}, checkpoint saved ✅")
        
        # Step 4: Verify all checkpoints
        print("4. 🔐 Verifying checkpoint integrity...")
        verification_results = verifier.verify_all_checkpoints()
        all_valid = all(verification_results.values())
        print(f"   All checkpoints valid: {all_valid} {'✅' if all_valid else '❌'}")
        
        # Step 5: Test reproducibility
        print("5. 🔄 Testing experiment reproducibility...")
        
        def run_experiment():
            set_global_seed(42)
            data = np.random.randn(50, 10)
            return np.mean(data @ np.random.randn(10, 1))
        
        results = [run_experiment() for _ in range(3)]
        all_identical = all(abs(r - results[0]) < 1e-15 for r in results[1:])
        print(f"   3 runs produce identical results: {all_identical} {'✅' if all_identical else '❌'}")
        print(f"   Sample result: {results[0]:.10f}")
        
        print("\n🎉 Complete workflow demonstration successful!")


def main():
    """Run all demonstrations."""
    print("🧪 ML Cookbook: Reproducibility Toolkit Demo")
    print("=" * 50)
    print("Project 1.2: The Reproducibility Toolkit")
    print("=" * 50)
    
    demos = [
        demo_seeding,
        demo_deterministic_operations,
        demo_checkpoint_verification,
        demo_project_templates,
        demo_complete_workflow
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            demo()
            if i < len(demos):
                print("\n" + "─" * 50)
        except Exception as e:
            print(f"❌ Demo {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎯 Key Takeaways:")
    print("  • Global seed management ensures reproducible random operations")
    print("  • Deterministic configurations eliminate non-deterministic behavior")
    print("  • Checkpoint verification prevents data corruption")
    print("  • Project templates provide reproducibility best practices")
    print("  • Complete workflow integration enables reliable experimentation")
    
    print("\n🚀 Ready to build reliable, reproducible ML systems!")


if __name__ == "__main__":
    main()
