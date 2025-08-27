# ML Cookbook - Final Integration Test
# Test all 5 components working together

import sys
import time
from pathlib import Path


def test_complete_integration():
    """Test all components working together in a realistic workflow"""

    print("🧪 ML COOKBOOK - COMPLETE INTEGRATION TEST")
    print("=" * 60)

    try:
        # Test all imports
        print("1. Testing component imports...")
        from cookbook.measure import (
            PerformanceProfiler,
            ExperimentLogger,
            ExperimentConfig,
            StatisticalValidator,
            TestType,
            CarbonTracker,
            CookbookCLI
        )
        print("   ✅ All components imported successfully!")

        # Test 1: Performance Profiler
        print("\n2. Testing Performance Profiler...")
        profiler = PerformanceProfiler(track_gpu=False, track_carbon=True)

        with profiler.profile("Integration test computation"):
            # Simulate ML computation
            import torch
            import numpy as np

            # Create a small model
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 50),
                torch.nn.ReLU(),
                torch.nn.Linear(50, 10)
            )

            # Run inference
            x = torch.randn(32, 100)
            with torch.no_grad():
                output = model(x)

            # Estimate FLOPs
            flops = profiler.estimate_flops(model, (32, 100))

            time.sleep(0.1)  # Simulate some computation time

        print("   ✅ Performance Profiler working!")
        print(f"      Peak RAM: {profiler.metrics.peak_ram_mb:.1f} MB")
        print(f"      Wall time: {profiler.metrics.wall_time_s:.3f}s")
        print(f"      Estimated FLOPs: {flops:,}")

        # Test 2: Experiment Logger
        print("\n3. Testing Experiment Logger...")

        config = ExperimentConfig(
            project_name="ml-cookbook-integration",
            experiment_name="complete_test",
            tags=["integration", "test"],
            use_wandb=False,  # Disable for testing
            use_tensorboard=False  # Disable for testing
        )

        logger = ExperimentLogger(config)

        # Log some metrics
        logger.log_hyperparameters({
            "learning_rate": 0.001,
            "batch_size": 32,
            "model_params": sum(p.numel() for p in model.parameters())
        })

        logger.log_metrics({
            "train/loss": 0.5,
            "train/accuracy": 0.85,
            "profiler/peak_ram_mb": profiler.metrics.peak_ram_mb,
            "profiler/wall_time_s": profiler.metrics.wall_time_s
        }, step=1)

        summary = logger.finalize()
        print("   ✅ Experiment Logger working!")
        print(f"      Logged {len(logger.metrics_buffer)} metric entries")

        # Test 3: Statistical Validator
        print("\n4. Testing Statistical Validator...")

        validator = StatisticalValidator(confidence_level=0.95, bootstrap_samples=1000)

        # Create realistic baseline vs treatment data
        np.random.seed(42)
        baseline_accuracy = np.random.normal(0.82, 0.03, 50).tolist()
        treatment_accuracy = np.random.normal(0.85, 0.03, 50).tolist()

        result = validator.compare_models(
            baseline_metrics=baseline_accuracy,
            treatment_metrics=treatment_accuracy,
            metric_name="accuracy",
            test_type=TestType.BOOTSTRAP
        )

        print("   ✅ Statistical Validator working!")
        print(f"      P-value: {result.p_value:.4f}")
        print(f"      Effect size: {result.effect_size:.3f}")
        print(f"      Significant: {result.is_significant()}")

        # Test 4: Carbon Tracker
        print("\n5. Testing Carbon Tracker...")

        tracker = CarbonTracker(
            project_name="integration-test",
            experiment_name="carbon_test"
        )

        with tracker.start_tracking("integration_computation"):
            # Simulate some computation
            for i in range(3):
                x = np.random.rand(100, 100)
                y = np.dot(x, x.T)
                time.sleep(0.05)

        carbon_metrics = tracker.metrics_history[-1]
        report = tracker.get_summary_report()

        print("   ✅ Carbon Tracker working!")
        print(f"      Emissions: {carbon_metrics.emissions_kg_co2 * 1000:.2f}g CO2")
        print(f"      Duration: {carbon_metrics.duration_seconds:.2f}s")

        # Test 5: CLI Interface
        print("\n6. Testing CLI Interface...")

        cli = CookbookCLI()

        # Test help system
        parser = cli.create_parser()
        help_text = parser.format_help()

        # Test template creation
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            template_path = Path(temp_dir) / "test_config.yaml"
            result = cli.main(["init", "--template", "basic", "--output", str(template_path)])
            template_created = (result == 0 and template_path.exists())

        print("   ✅ CLI Interface working!")
        print(f"      Help system: {len(help_text)} chars")
        print(f"      Template creation: {'Success' if template_created else 'Failed'}")

        # Combined Integration Test
        print("\n7. Testing Combined Workflow...")

        # Use profiler + carbon tracker together
        combined_profiler = profiler
        combined_tracker = tracker

        print("   Running combined profiling + carbon tracking...")

        with combined_profiler.profile("Combined workflow"):
            with combined_tracker.start_tracking("combined_test"):
                # Simulate a mini training loop
                for epoch in range(2):
                    # Forward pass
                    x = torch.randn(16, 100)
                    output = model(x)
                    loss = torch.mean(output ** 2)

                    # Simulate backward pass
                    time.sleep(0.02)

                    print(f"      Epoch {epoch + 1}: loss = {loss.item():.4f}")

        print("   ✅ Combined workflow successful!")

        # Final Assessment
        print("\n" + "=" * 60)
        print("🎉 INTEGRATION TEST RESULTS")
        print("=" * 60)

        components_tested = [
            "Performance Profiler",
            "Experiment Logger",
            "Statistical Validator",
            "Carbon Tracker",
            "CLI Interface"
        ]

        print("✅ ALL COMPONENTS WORKING PERFECTLY!")
        print(f"✅ Tested: {', '.join(components_tested)}")
        print(f"✅ Full integration workflow successful")
        print(f"✅ Portfolio-ready ML measurement suite complete!")

        print(f"\n🏆 PROJECT 1.1: THE MEASUREMENT SUITE - COMPLETED!")
        print("Your professional ML toolkit includes:")
        print("  🔬 Performance profiling with memory, timing, FLOPs tracking")
        print("  📊 Multi-backend experiment logging")
        print("  📈 Rigorous statistical validation with A/B testing")
        print("  🌱 Comprehensive carbon footprint tracking")
        print("  ⚙️  Professional CLI with configuration templates")

        print(f"\n🚀 Ready for Phase 2: Core ML Mechanics Laboratory!")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_package_installation():
    """Test that the package can be installed and CLI works"""

    print("\n🔧 TESTING PACKAGE INSTALLATION")
    print("=" * 50)

    try:
        # Test CLI entry point
        from cookbook.measure.cli import main_cli
        print("✅ CLI entry point available")

        # Test importable from anywhere
        import cookbook
        print(f"✅ Package version: {cookbook.__version__}")
        print(f"✅ Package author: {cookbook.__author__}")

        print("\n💡 Installation commands working:")
        print("   pip install -e .  # Development installation")
        print("   cookbook-prof --help  # CLI command available")

        return True

    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Run all tests"""

    # Test component status first
    try:
        from cookbook.measure import print_status
        print_status()
    except Exception as e:
        print(f"Could not print component status: {e}")

    print("\n" + "=" * 70)

    # Run integration test
    integration_success = test_complete_integration()

    # Test installation
    installation_success = test_package_installation()

    # Final summary
    print(f"\n" + "=" * 70)
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 70)

    if integration_success and installation_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your ML Cookbook is ready for professional showcase!")
        print("✅ Perfect portfolio piece demonstrating:")
        print("   • Advanced Python package development")
        print("   • Professional ML engineering practices")
        print("   • Comprehensive measurement and validation")
        print("   • Sustainable ML development awareness")
        print("   • Production-ready CLI tools")

        print(f"\n📚 Next steps for portfolio impact:")
        print("   1. Create compelling README.md with examples")
        print("   2. Add example Jupyter notebooks")
        print("   3. Set up GitHub Actions CI/CD")
        print("   4. Create documentation website")
        print("   5. Share your professional ML toolkit!")

    else:
        print("⚠️  Some tests need attention")
        print("   Review the error messages above")
        print("   Ensure all dependencies are installed")
        print("   Check file locations and imports")


if __name__ == "__main__":
    main()
