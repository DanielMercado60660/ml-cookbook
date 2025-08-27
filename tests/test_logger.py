# Project 1.1: Experiment Logger Validation
# Test the logger functionality and validate against Quality Gates

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import cookbook.measure.logger


def validate_logger_functionality():
    """Validate logger meets Quality Gates"""
    print("🔬 EXPERIMENT LOGGER VALIDATION")
    print("=" * 50)

    results = {}

    # Test 1: Basic functionality
    print("📝 Test 1: Basic logging functionality...")

    try:
        logger = cookbook.measure.logger.create_logger(
            project_name="ml-cookbook-validation",
            experiment_name="logger_test",
            tags=["validation", "test"],
            log_dir="./logs"
        )

        # Test hyperparameter logging
        test_hparams = {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10
        }
        logger.log_hyperparameters(test_hparams)

        # Test metric logging
        for i in range(5):
            metrics = {
                "loss": 1.0 - i * 0.1,
                "accuracy": i * 0.2,
                "step_time": 0.5 + np.random.normal(0, 0.05)
            }
            logger.log_metrics(metrics, step=i)

        # Test section timing
        with logger.log_section("test_section"):
            import time
            time.sleep(0.1)  # Small delay to test timing

        summary = logger.finalize()

        results['basic_functionality'] = {
            'passed': True,
            'metrics_logged': len(logger.metrics_buffer),
            'summary_keys': list(summary.keys())
        }

        print("   ✅ Basic functionality test passed")

    except Exception as e:
        results['basic_functionality'] = {
            'passed': False,
            'error': str(e)
        }
        print(f"   ❌ Basic functionality test failed: {e}")

    # Test 2: Local file creation and format
    print("\n📁 Test 2: Local file validation...")

    try:
        # Check if JSONL file was created
        log_files = list(Path("./logs").glob("*/metrics.jsonl"))

        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)

            # Validate JSONL format
            valid_lines = 0
            total_lines = 0

            with open(latest_log, 'r') as f:
                for line in f:
                    total_lines += 1
                    try:
                        json.loads(line.strip())
                        valid_lines += 1
                    except json.JSONDecodeError:
                        pass

            format_validity = (valid_lines / total_lines) * 100 if total_lines > 0 else 0

            results['local_file_validation'] = {
                'passed': format_validity >= 95.0,
                'file_exists': True,
                'total_lines': total_lines,
                'valid_lines': valid_lines,
                'format_validity_percent': format_validity,
                'file_path': str(latest_log)
            }

            if format_validity >= 95.0:
                print(f"   ✅ Local file validation passed ({format_validity:.1f}% valid)")
                print(f"   📄 Log file: {latest_log}")
            else:
                print(f"   ❌ Local file validation failed ({format_validity:.1f}% valid)")

        else:
            results['local_file_validation'] = {
                'passed': False,
                'file_exists': False,
                'error': "No log files found"
            }
            print("   ❌ No log files found")

    except Exception as e:
        results['local_file_validation'] = {
            'passed': False,
            'error': str(e)
        }
        print(f"   ❌ Local file validation failed: {e}")

    # Test 3: Backend availability check
    print("\n🔌 Test 3: Backend availability...")

    backend_status = {
        'wandb_available': cookbook.measure.logger.WANDB_AVAILABLE,
        'tensorboard_available': cookbook.measure.logger.TENSORBOARD_AVAILABLE,
        'local_always_available': True
    }

    results['backend_availability'] = backend_status

    for backend, available in backend_status.items():
        status = "✅" if available else "❌"
        print(f"   {status} {backend}: {'Available' if available else 'Not available'}")

    # Overall assessment
    print("\n" + "=" * 50)

    basic_passed = results.get('basic_functionality', {}).get('passed', False)
    file_passed = results.get('local_file_validation', {}).get('passed', False)
    local_available = backend_status['local_always_available']

    overall_passed = basic_passed and file_passed and local_available

    results['overall_assessment'] = {
        'passed': overall_passed,
        'basic_functionality': basic_passed,
        'file_validation': file_passed,
        'local_backend': local_available
    }

    if overall_passed:
        print("🎯 OVERALL VALIDATION RESULT:")
        print("   ✅ EXPERIMENT LOGGER MEETS QUALITY GATES")
        print("   Ready for portfolio integration!")
    else:
        print("🎯 OVERALL VALIDATION RESULT:")
        print("   ❌ EXPERIMENT LOGGER NEEDS IMPROVEMENTS")
        failed_tests = []
        if not basic_passed:
            failed_tests.append("basic functionality")
        if not file_passed:
            failed_tests.append("file validation")
        print(f"   Failed tests: {', '.join(failed_tests)}")

    return results


def visualize_logger_demo():
    """Create portfolio-quality visualization of logger demo"""
    print("\n📊 Creating logger demonstration...")

    # Run the demo to get data
    logger, summary = cookbook.measure.logger.demo_logger()

    # Extract metrics for plotting
    metrics_data = {}
    for entry in logger.metrics_buffer:
        for key, value in entry['metrics'].items():
            if key not in metrics_data:
                metrics_data[key] = {'steps': [], 'values': []}
            metrics_data[key]['steps'].append(entry['step'])
            metrics_data[key]['values'].append(value)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    if 'train/loss' in metrics_data:
        ax = axes[0, 0]
        data = metrics_data['train/loss']
        ax.plot(data['steps'], data['values'], 'o-', color='red', linewidth=2, markersize=6)
        ax.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    # Plot 2: Training Accuracy
    if 'train/accuracy' in metrics_data:
        ax = axes[0, 1]
        data = metrics_data['train/accuracy']
        ax.plot(data['steps'], data['values'], 'o-', color='green', linewidth=2, markersize=6)
        ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Accuracy')
        ax.grid(True, alpha=0.3)

    # Plot 3: Learning Rate Decay
    if 'train/learning_rate' in metrics_data:
        ax = axes[1, 0]
        data = metrics_data['train/learning_rate']
        ax.plot(data['steps'], data['values'], 'o-', color='blue', linewidth=2, markersize=6)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    # Plot 4: Validation vs Training Loss
    ax = axes[1, 1]
    if 'train/loss' in metrics_data and 'val/loss' in metrics_data:
        train_data = metrics_data['train/loss']
        val_data = metrics_data['val/loss']

        ax.plot(train_data['steps'], train_data['values'], 'o-',
                color='red', label='Training', linewidth=2, markersize=6)
        ax.plot(val_data['steps'], val_data['values'], 's-',
                color='orange', label='Validation', linewidth=2, markersize=6)
        ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('ML Cookbook - Experiment Logger Demo', y=1.02, fontsize=16, fontweight='bold')
    plt.show()

    return fig


# Main validation runner
def run_logger_validation():
    """Run complete logger validation suite"""

    # Run validation tests
    validation_results = validate_logger_functionality()

    # Create demo visualization
    try:
        demo_fig = visualize_logger_demo()
    except Exception as e:
        print(f"⚠️  Visualization creation failed: {e}")
        demo_fig = None

    # Save validation report
    report_path = "./logs/logger_validation.json"
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)

    print(f"\n💾 Logger validation report saved: {report_path}")

    return validation_results, demo_fig


# Execute validation
print("🚀 Starting Experiment Logger Validation...")
logger_results, demo_figure = run_logger_validation()