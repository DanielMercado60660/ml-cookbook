# Project 1.1: The Measurement Suite - Performance Profiler
# Portfolio-quality toolkit for analyzing model performance

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import threading


@dataclass
class ProfilerMetrics:
    """Container for profiling metrics"""
    # Memory metrics (MB)
    peak_ram_mb: float = 0.0
    step_ram_mb: List[float] = field(default_factory=list)
    gpu_memory_mb: Optional[float] = None

    # Timing metrics (seconds)
    wall_time_s: float = 0.0
    step_times_s: List[float] = field(default_factory=list)

    # Compute metrics
    estimated_flops: int = 0
    throughput_samples_per_sec: float = 0.0

    # Sustainability metrics
    energy_consumption_kwh: float = 0.0
    carbon_footprint_kg: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            'memory': {
                'peak_ram_mb': self.peak_ram_mb,
                'avg_step_ram_mb': np.mean(self.step_ram_mb) if self.step_ram_mb else 0.0,
                'gpu_memory_mb': self.gpu_memory_mb
            },
            'timing': {
                'wall_time_s': self.wall_time_s,
                'avg_step_time_s': np.mean(self.step_times_s) if self.step_times_s else 0.0,
                'throughput_samples_per_sec': self.throughput_samples_per_sec
            },
            'compute': {
                'estimated_flops': self.estimated_flops
            },
            'sustainability': {
                'energy_consumption_kwh': self.energy_consumption_kwh,
                'carbon_footprint_kg': self.carbon_footprint_kg
            }
        }


class PerformanceProfiler:
    """
    Production-quality performance profiler for ML workloads

    Tracks memory usage, timing, FLOPs, and sustainability metrics
    with sub-1% accuracy targets for portfolio validation.
    """

    def __init__(self,
                 track_gpu: bool = True,
                 track_carbon: bool = True,
                 carbon_intensity_g_per_kwh: float = 429.0):  # Global average
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.track_carbon = track_carbon
        self.carbon_intensity = carbon_intensity_g_per_kwh

        self.metrics = ProfilerMetrics()
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = None

        # Initialize process monitoring
        self.process = psutil.Process()

        print(f"🔬 Performance Profiler initialized")
        print(f"   GPU tracking: {'✅' if self.track_gpu else '❌'}")
        print(f"   Carbon tracking: {'✅' if self.track_carbon else '❌'}")

    def _monitor_memory(self):
        """Background thread to monitor memory usage"""
        while self._monitoring:
            # System RAM
            ram_mb = self.process.memory_info().rss / 1024 / 1024
            self.metrics.step_ram_mb.append(ram_mb)
            self.metrics.peak_ram_mb = max(self.metrics.peak_ram_mb, ram_mb)

            # GPU memory
            if self.track_gpu:
                gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                if self.metrics.gpu_memory_mb is None:
                    self.metrics.gpu_memory_mb = gpu_mem
                else:
                    self.metrics.gpu_memory_mb = max(self.metrics.gpu_memory_mb, gpu_mem)

            time.sleep(0.1)  # 10Hz monitoring

    @contextmanager
    def profile(self, description: str = ""):
        """Context manager for profiling code blocks"""
        print(f"🚀 Starting profiling: {description}")

        # Start monitoring
        self._start_time = time.perf_counter()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()

        # Clear GPU cache if available
        if self.track_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        try:
            yield self.metrics
        finally:
            # Stop monitoring
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)

            # Record final metrics
            self.metrics.wall_time_s = time.perf_counter() - self._start_time

            if self.track_gpu:
                self.metrics.gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            self._estimate_energy_consumption()

            print(f"✅ Profiling complete: {description}")
            self._print_summary()

    def profile_step(self, step_func: Callable, *args, **kwargs):
        """Profile a single training/inference step"""
        step_start = time.perf_counter()

        # Record pre-step memory
        pre_ram = self.process.memory_info().rss / 1024 / 1024

        # Execute step
        result = step_func(*args, **kwargs)

        # Record timing
        step_time = time.perf_counter() - step_start
        self.metrics.step_times_s.append(step_time)

        # Record post-step memory
        post_ram = self.process.memory_info().rss / 1024 / 1024
        self.metrics.step_ram_mb.append(post_ram)
        self.metrics.peak_ram_mb = max(self.metrics.peak_ram_mb, post_ram)

        return result

    def estimate_flops(self, model: torch.nn.Module,
                       input_shape: tuple,
                       batch_size: int = 1) -> int:
        """
        Estimate FLOPs for a model given input shape
        Note: This is approximate - document assumptions clearly
        """
        total_flops = 0

        def count_conv2d_flops(module, input_shape):
            # Simplified FLOP counting for Conv2d
            kernel_flops = np.prod(module.kernel_size) * module.in_channels
            output_elements = np.prod(input_shape[2:]) * module.out_channels
            return kernel_flops * output_elements

        def count_linear_flops(module, input_size):
            return module.in_features * module.out_features

        # Walk through model and estimate FLOPs
        # This is a simplified version - production would use torch.profiler
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                total_flops += count_conv2d_flops(module, input_shape)
            elif isinstance(module, torch.nn.Linear):
                total_flops += count_linear_flops(module, input_shape[-1])

        self.metrics.estimated_flops = total_flops * batch_size
        return self.metrics.estimated_flops

    def _estimate_energy_consumption(self):
        """Estimate energy consumption based on compute time and hardware"""
        if not self.track_carbon or self.metrics.wall_time_s == 0:
            return

        # Rough estimates - would use CodeCarbon in production
        if self.track_gpu:
            # Assume ~300W for GPU + ~100W for system
            power_watts = 400.0
        else:
            # CPU-only system
            power_watts = 100.0

        # Calculate energy consumption
        energy_kwh = (power_watts * self.metrics.wall_time_s / 3600) / 1000
        self.metrics.energy_consumption_kwh = energy_kwh

        # Estimate carbon footprint
        self.metrics.carbon_footprint_kg = energy_kwh * self.carbon_intensity / 1000

    def _print_summary(self):
        """Print a formatted summary of profiling results"""
        print("\n" + "=" * 60)
        print("📊 PROFILING SUMMARY")
        print("=" * 60)

        # Memory
        print(f"🧠 Memory:")
        print(f"   Peak RAM: {self.metrics.peak_ram_mb:.1f} MB")
        if self.metrics.gpu_memory_mb:
            print(f"   Peak GPU: {self.metrics.gpu_memory_mb:.1f} MB")

        # Timing
        print(f"⏱️  Timing:")
        print(f"   Wall time: {self.metrics.wall_time_s:.3f}s")
        if self.metrics.step_times_s:
            avg_step = np.mean(self.metrics.step_times_s) * 1000
            print(f"   Avg step: {avg_step:.2f}ms")

        # Compute
        if self.metrics.estimated_flops > 0:
            print(f"💻 Compute:")
            print(f"   Est. FLOPs: {self.metrics.estimated_flops:,}")
            gflops_per_sec = self.metrics.estimated_flops / self.metrics.wall_time_s / 1e9
            print(f"   GFLOP/s: {gflops_per_sec:.2f}")

        # Sustainability
        if self.track_carbon and self.metrics.energy_consumption_kwh > 0:
            print(f"🌱 Sustainability:")
            print(f"   Energy: {self.metrics.energy_consumption_kwh * 1000:.2f} Wh")
            print(f"   Carbon: {self.metrics.carbon_footprint_kg * 1000:.2f}g CO₂eq")

        print("=" * 60)

    def save_metrics(self, filepath: str):
        """Save metrics to JSON for later analysis"""
        metrics_dict = self.metrics.to_dict()
        metrics_dict['metadata'] = {
            'timestamp': time.time(),
            'gpu_available': self.track_gpu,
            'carbon_tracking': self.track_carbon
        }

        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"💾 Metrics saved to: {filepath}")


# Demo usage and validation
def demo_profiler():
    """Demo the profiler with a simple neural network"""
    print("🧪 PROFILER DEMO - Creating a simple model for testing...")

    # Create a simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )

    # Create dummy data
    batch_size = 32
    x = torch.randn(batch_size, 784)
    target = torch.randint(0, 10, (batch_size,))

    # Initialize profiler
    profiler = PerformanceProfiler()

    # Profile model inference
    with profiler.profile("Model inference"):
        # Estimate FLOPs
        profiler.estimate_flops(model, (batch_size, 784))

        # Run inference
        with torch.no_grad():
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, target)

    # Save results
    profiler.save_metrics("/tmp/demo_metrics.json")

    return profiler


# Run the demo
if __name__ == "__main__":
    demo_profiler()