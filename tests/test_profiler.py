# Project 1.1 Validation Suite
# Verify profiler meets ±1-2% accuracy and ±5% wall-time targets

import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import json

from cookbook import PerformanceProfiler


class ProfilerValidator:
    """Validates profiler accuracy against known baselines"""

    def __init__(self, profiler_class):
        self.profiler_class = profiler_class
        self.baselines = {}

    def create_known_workload(self, size: str = "small"):
        """Create predictable workloads for validation"""
        workloads = {
            "small": {
                "model": torch.nn.Sequential(
                    torch.nn.Linear(100, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 10)
                ),
                "input_shape": (32, 100),
                "expected_params": 100 * 50 + 50 + 50 * 10 + 10,  # 5610 params
                "min_wall_time_ms": 1.0,  # Very rough baseline
            },
            "medium": {
                "model": torch.nn.Sequential(
                    torch.nn.Linear(784, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 10)
                ),
                "input_shape": (64, 784),
                "expected_params": 784 * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10,
                "min_wall_time_ms": 5.0,
            }
        }
        return workloads[size]

    def benchmark_timing_consistency(self, n_runs: int = 5) -> Dict:
        """Test timing consistency across multiple runs"""
        print(f"🎯 Running timing consistency test ({n_runs} runs)...")

        workload = self.create_known_workload("medium")
        model = workload["model"]
        input_shape = workload["input_shape"]

        results = []

        for run in range(n_runs):
            profiler = self.profiler_class()
            x = torch.randn(*input_shape)

            with profiler.profile(f"Consistency test run {run + 1}"):
                with torch.no_grad():
                    _ = model(x)

            results.append({
                'wall_time_s': profiler.metrics.wall_time_s,
                'peak_ram_mb': profiler.metrics.peak_ram_mb,
                'run': run
            })

            print(f"   Run {run + 1}: {profiler.metrics.wall_time_s * 1000:.2f}ms, "
                  f"{profiler.metrics.peak_ram_mb:.1f}MB RAM")

        # Calculate statistics
        wall_times = [r['wall_time_s'] for r in results]
        ram_peaks = [r['peak_ram_mb'] for r in results]

        timing_stats = {
            'mean_ms': np.mean(wall_times) * 1000,
            'std_ms': np.std(wall_times) * 1000,
            'cv_percent': (np.std(wall_times) / np.mean(wall_times)) * 100,
            'ram_consistency_mb': np.std(ram_peaks)
        }

        print(f"\n📊 Timing Consistency Results:")
        print(f"   Mean: {timing_stats['mean_ms']:.2f}ms ± {timing_stats['std_ms']:.2f}ms")
        print(f"   Coefficient of Variation: {timing_stats['cv_percent']:.1f}%")
        print(f"   RAM Consistency: ±{timing_stats['ram_consistency_mb']:.1f}MB")

        # Check quality gate
        if timing_stats['cv_percent'] <= 5.0:
            print("   ✅ PASS: Timing consistency within ±5%")
        else:
            print("   ❌ FAIL: Timing consistency exceeds ±5%")

        return {
            'results': results,
            'stats': timing_stats,
            'passed_timing_gate': timing_stats['cv_percent'] <= 5.0
        }

    def benchmark_memory_accuracy(self) -> Dict:
        """Test memory tracking accuracy"""
        print("🧠 Testing memory tracking accuracy...")

        profiler = self.profiler_class()

        # Create known memory allocation
        initial_ram = profiler.process.memory_info().rss / 1024 / 1024

        with profiler.profile("Memory accuracy test"):
            # Allocate ~100MB tensor
            big_tensor = torch.randn(1000, 25000)  # ~100MB

            # Force some computation to ensure allocation
            result = torch.sum(big_tensor ** 2)

        # Check if profiler caught the allocation
        memory_increase = profiler.metrics.peak_ram_mb - initial_ram
        expected_increase = 100  # MB

        accuracy_percent = abs(memory_increase - expected_increase) / expected_increase * 100

        print(f"   Expected increase: ~{expected_increase}MB")
        print(f"   Measured increase: {memory_increase:.1f}MB")
        print(f"   Accuracy error: {accuracy_percent:.1f}%")

        passed = accuracy_percent <= 10.0  # Allow 10% error for memory tracking

        if passed:
            print("   ✅ PASS: Memory tracking within acceptable range")
        else:
            print("   ❌ FAIL: Memory tracking error too high")

        return {
            'memory_increase_mb': memory_increase,
            'accuracy_error_percent': accuracy_percent,
            'passed_memory_gate': passed
        }

    def run_full_validation(self) -> Dict:
        """Run complete validation suite"""
        print("🔬 PROFILER VALIDATION SUITE")
        print("=" * 50)

        results = {}

        # Test 1: Timing consistency
        results['timing'] = self.benchmark_timing_consistency()

        print("\n" + "-" * 50)

        # Test 2: Memory accuracy
        results['memory'] = self.benchmark_memory_accuracy()

        print("\n" + "-" * 50)

        # Overall assessment
        all_passed = (
                results['timing']['passed_timing_gate'] and
                results['memory']['passed_memory_gate']
        )

        print(f"\n🎯 OVERALL VALIDATION RESULT:")
        if all_passed:
            print("   ✅ PROFILER MEETS QUALITY GATES")
            print("   Ready for portfolio integration!")
        else:
            print("   ❌ PROFILER NEEDS IMPROVEMENTS")
            print("   Review failed tests before proceeding")

        results['overall_passed'] = all_passed
        return results

    def visualize_results(self, results: Dict):
        """Create portfolio-quality visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Timing consistency
        timing_data = results['timing']['results']
        wall_times_ms = [r['wall_time_s'] * 1000 for r in timing_data]
        runs = [r['run'] + 1 for r in timing_data]

        ax1.plot(runs, wall_times_ms, 'o-', color='blue', markersize=8)
        ax1.set_title('Timing Consistency Across Runs')
        ax1.set_xlabel('Run Number')
        ax1.set_ylabel('Wall Time (ms)')
        ax1.grid(True, alpha=0.3)

        # Add mean line
        mean_time = np.mean(wall_times_ms)
        ax1.axhline(y=mean_time, color='red', linestyle='--', alpha=0.7,
                    label=f'Mean: {mean_time:.2f}ms')
        ax1.legend()

        # Plot 2: Memory tracking
        ram_values = [r['peak_ram_mb'] for r in timing_data]
        ax2.bar(runs, ram_values, color='green', alpha=0.7)
        ax2.set_title('Peak RAM Usage Per Run')
        ax2.set_xlabel('Run Number')
        ax2.set_ylabel('Peak RAM (MB)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Coefficient of Variation
        cv = results['timing']['stats']['cv_percent']
        colors = ['green' if cv <= 5 else 'red']
        ax3.bar(['Timing CV'], [cv], color=colors, alpha=0.7)
        ax3.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='5% Target')
        ax3.set_title('Timing Coefficient of Variation')
        ax3.set_ylabel('CV (%)')
        ax3.legend()
        ax3.set_ylim(0, max(10, cv + 1))

        # Plot 4: Memory accuracy
        mem_error = results['memory']['accuracy_error_percent']
        colors = ['green' if mem_error <= 10 else 'red']
        ax4.bar(['Memory Error'], [mem_error], color=colors, alpha=0.7)
        ax4.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='10% Target')
        ax4.set_title('Memory Tracking Accuracy')
        ax4.set_ylabel('Error (%)')
        ax4.legend()
        ax4.set_ylim(0, max(15, mem_error + 1))

        plt.tight_layout()
        plt.suptitle('ML Cookbook - Profiler Validation Results', y=1.02, fontsize=16)
        plt.show()

        return fig


# Run the validation
def run_profiler_validation():
    """Execute the full profiler validation suite"""

    # Import your profiler class (assumes it's already loaded)
    validator = ProfilerValidator(PerformanceProfiler)

    # Run validation
    results = validator.run_full_validation()

    # Create visualizations
    validator.visualize_results(results)

    # Save validation report
    import os
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    validation_path = os.path.join(log_dir, 'profiler_validation.json')
    with open(validation_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Validation report saved to: {validation_path}")

    return results


# Run validation when this cell executes
if __name__ == "__main__":
    print("Starting validation now...")
    validation_results = run_profiler_validation()