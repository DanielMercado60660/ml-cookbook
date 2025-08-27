# Performance Profiler

The PerformanceProfiler provides comprehensive system resource and compute tracking for ML workloads.

## Overview

The Performance Profiler tracks:
- **Memory Usage**: Peak RAM, GPU memory, memory leaks
- **Timing**: Wall time, CPU time, per-operation timing
- **Compute**: FLOPS estimation, throughput analysis
- **System**: CPU/GPU utilization, temperature monitoring

## Basic Usage

### Simple Profiling

```python
from cookbook.measure import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("training_loop"):
    # Your ML code here
    model.fit(X_train, y_train)

# Get comprehensive metrics
metrics = profiler.get_metrics()
print(f"Peak RAM: {metrics.memory.peak_ram_mb:.1f} MB")
print(f"Wall Time: {metrics.timing.wall_time_s:.3f}s")
```

### Advanced Configuration

```python
profiler = PerformanceProfiler(
    track_gpu=True,           # Enable GPU monitoring
    track_carbon=False,       # Disable carbon tracking for pure performance
    detailed_memory=True,     # Detailed memory analysis
    memory_interval=0.1,      # Sample memory every 100ms
    track_cpu_temp=True,      # Monitor CPU temperature
    track_network=False       # Disable network monitoring
)
```

## Memory Profiling

### Memory Leak Detection

```python
profiler = PerformanceProfiler(detailed_memory=True)

# Monitor memory over time
memory_snapshots = []

with profiler.profile("potential_memory_leak"):
    for epoch in range(10):
        # Training code that might leak memory
        train_epoch()
        
        # Take memory snapshot
        current_memory = profiler.get_current_memory()
        memory_snapshots.append(current_memory)

# Analyze memory growth
memory_growth = memory_snapshots[-1] - memory_snapshots[0]
if memory_growth > 100:  # More than 100MB growth
    print(f"⚠️ Potential memory leak: {memory_growth:.1f}MB growth")
```

### GPU Memory Tracking

```python
import torch

profiler = PerformanceProfiler(track_gpu=True)

with profiler.profile("gpu_training"):
    model = MyModel().cuda()
    
    for batch in dataloader:
        output = model(batch.cuda())
        loss = criterion(output, target.cuda())
        
        # Check GPU memory at critical points
        gpu_memory = profiler.get_current_gpu_memory()
        if gpu_memory > 0.9:  # >90% GPU memory usage
            print("⚠️ High GPU memory usage, consider reducing batch size")

metrics = profiler.get_metrics()
print(f"Peak GPU Memory: {metrics.memory.peak_gpu_mb:.1f} MB")
print(f"GPU Utilization: {metrics.compute.gpu_utilization_percent:.1f}%")
```

## Timing Analysis

### Detailed Timing Breakdown

```python
profiler = PerformanceProfiler()

with profiler.profile("complete_pipeline"):
    # Profile individual components
    with profiler.profile("data_loading"):
        data = load_data()
    
    with profiler.profile("preprocessing"):
        data = preprocess(data)
    
    with profiler.profile("training"):
        model = train_model(data)
    
    with profiler.profile("evaluation"):
        results = evaluate(model)

# Get timing breakdown
timing_report = profiler.get_timing_breakdown()
for operation, timing in timing_report.items():
    print(f"{operation}: {timing:.3f}s ({timing/sum(timing_report.values())*100:.1f}%)")
```

### Performance Bottleneck Detection

```python
profiler = PerformanceProfiler(enable_line_profiling=True)

with profiler.profile("bottleneck_detection"):
    # Code with potential bottlenecks
    slow_function()

# Get line-by-line timing
line_profile = profiler.get_line_profile()
bottlenecks = [line for line in line_profile if line.time_percent > 10]

for bottleneck in bottlenecks:
    print(f"Bottleneck at line {bottleneck.line_number}: {bottleneck.time_percent:.1f}% of time")
```

## FLOPS Estimation

### Model Complexity Analysis

```python
import torch
import torch.nn as nn

profiler = PerformanceProfiler(estimate_flops=True)

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

with profiler.profile("model_complexity"):
    # Forward pass for FLOPS estimation
    x = torch.randn(32, 784)
    output = model(x)

metrics = profiler.get_metrics()
print(f"Estimated FLOPs: {metrics.compute.estimated_flops:,}")
print(f"FLOPs per parameter: {metrics.compute.flops_per_param:.2f}")
```

### Throughput Analysis

```python
profiler = PerformanceProfiler()

batch_sizes = [16, 32, 64, 128]
throughput_results = {}

for batch_size in batch_sizes:
    with profiler.profile(f"batch_size_{batch_size}"):
        # Process multiple batches
        for _ in range(10):
            x = torch.randn(batch_size, 784)
            output = model(x)
    
    metrics = profiler.get_metrics()
    samples_per_second = (10 * batch_size) / metrics.timing.wall_time_s
    throughput_results[batch_size] = samples_per_second

# Find optimal batch size
optimal_batch = max(throughput_results, key=throughput_results.get)
print(f"Optimal batch size: {optimal_batch} ({throughput_results[optimal_batch]:.1f} samples/s)")
```

## Comparative Profiling

### Model Architecture Comparison

```python
profiler = PerformanceProfiler()

architectures = {
    "simple_mlp": SimpleMLPModel(),
    "deep_mlp": DeepMLPModel(),
    "cnn": CNNModel()
}

results = {}

for name, model in architectures.items():
    with profiler.profile(f"{name}_training"):
        # Standardized training loop
        train_model(model, train_data, epochs=5)
    
    metrics = profiler.get_metrics()
    results[name] = {
        "memory": metrics.memory.peak_ram_mb,
        "time": metrics.timing.wall_time_s,
        "flops": metrics.compute.estimated_flops
    }

# Display comparison
import pandas as pd
comparison_df = pd.DataFrame(results).T
print(comparison_df)
```

### Hyperparameter Impact Analysis

```python
profiler = PerformanceProfiler()

learning_rates = [0.001, 0.01, 0.1]
performance_impact = {}

for lr in learning_rates:
    with profiler.profile(f"lr_{lr}"):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_with_optimizer(model, optimizer)
    
    metrics = profiler.get_metrics()
    performance_impact[lr] = {
        "convergence_time": metrics.timing.wall_time_s,
        "memory_usage": metrics.memory.peak_ram_mb,
        "final_accuracy": get_final_accuracy()
    }

# Analyze efficiency vs accuracy tradeoff
for lr, impact in performance_impact.items():
    efficiency = impact["final_accuracy"] / impact["convergence_time"]
    print(f"LR {lr}: Efficiency = {efficiency:.4f} acc/second")
```

## System Monitoring

### Resource Utilization Tracking

```python
profiler = PerformanceProfiler(
    track_system=True,
    system_interval=1.0  # Sample system metrics every second
)

with profiler.profile("resource_monitoring"):
    # Long-running training process
    for epoch in range(100):
        train_epoch()
        
        # Get real-time system stats
        system_stats = profiler.get_current_system_stats()
        
        # Alert if system under stress
        if system_stats.cpu_percent > 90:
            print("⚠️ High CPU usage")
        if system_stats.memory_percent > 85:
            print("⚠️ High memory usage")
        if system_stats.disk_io_percent > 80:
            print("⚠️ High disk I/O")

# Get comprehensive system report
system_report = profiler.get_system_report()
print(f"Average CPU usage: {system_report.avg_cpu_percent:.1f}%")
print(f"Peak memory usage: {system_report.peak_memory_percent:.1f}%")
```

## Production Integration

### Automated Performance Regression Detection

```python
class PerformanceRegressionDetector:
    def __init__(self, baseline_metrics_path):
        self.baseline = self.load_baseline(baseline_metrics_path)
        self.profiler = PerformanceProfiler()
    
    def check_regression(self, operation_name, threshold=0.2):
        with self.profiler.profile(operation_name):
            # Run the operation
            yield
        
        current_metrics = self.profiler.get_metrics()
        baseline_time = self.baseline[operation_name]["wall_time_s"]
        
        if current_metrics.timing.wall_time_s > baseline_time * (1 + threshold):
            self.alert_regression(operation_name, baseline_time, current_metrics.timing.wall_time_s)

# Usage in CI/CD
detector = PerformanceRegressionDetector("baseline_metrics.json")

with detector.check_regression("model_training"):
    train_model()
```

### Production Monitoring Dashboard

```python
from datetime import datetime, timedelta
import json

class ProductionProfiler:
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.metrics_log = []
    
    def profile_request(self, request_id):
        return self.profiler.profile(f"request_{request_id}")
    
    def log_metrics(self, request_id, user_facing_latency):
        metrics = self.profiler.get_metrics()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "internal_latency": metrics.timing.wall_time_s,
            "user_facing_latency": user_facing_latency,
            "memory_mb": metrics.memory.peak_ram_mb,
            "cpu_percent": metrics.system.cpu_percent if hasattr(metrics, 'system') else None
        }
        
        self.metrics_log.append(log_entry)
        
        # Check SLA compliance
        if user_facing_latency > 2.0:  # 2 second SLA
            self.alert_sla_violation(log_entry)
    
    def generate_hourly_report(self):
        # Generate performance report for last hour
        cutoff = datetime.now() - timedelta(hours=1)
        recent_metrics = [m for m in self.metrics_log 
                         if datetime.fromisoformat(m["timestamp"]) > cutoff]
        
        if recent_metrics:
            avg_latency = sum(m["user_facing_latency"] for m in recent_metrics) / len(recent_metrics)
            p95_latency = sorted([m["user_facing_latency"] for m in recent_metrics])[int(0.95 * len(recent_metrics))]
            
            return {
                "requests_processed": len(recent_metrics),
                "average_latency": avg_latency,
                "p95_latency": p95_latency,
                "sla_violations": sum(1 for m in recent_metrics if m["user_facing_latency"] > 2.0)
            }

# Usage in production
prod_profiler = ProductionProfiler()

@app.route("/predict")
def predict():
    request_id = generate_request_id()
    
    start_time = time.time()
    with prod_profiler.profile_request(request_id):
        prediction = model.predict(request.json)
    end_time = time.time()
    
    prod_profiler.log_metrics(request_id, end_time - start_time)
    return {"prediction": prediction}
```

## Best Practices

### 1. Profile Representative Workloads

```python
# Profile with realistic data sizes and distributions
profiler = PerformanceProfiler()

# Don't profile toy examples
# ❌ with profiler.profile("toy_example"):
#     tiny_data = torch.randn(10, 10)
#     model(tiny_data)

# ✅ Profile with production-like data
with profiler.profile("realistic_workload"):
    production_batch = load_production_sample()
    model(production_batch)
```

### 2. Use Warm-up Runs

```python
profiler = PerformanceProfiler()

# Warm up the model (compilation, cache loading, etc.)
for _ in range(3):
    model(sample_input)

# Now profile the actual performance
with profiler.profile("warmed_up_inference"):
    for _ in range(100):
        model(sample_input)
```

### 3. Profile Both Training and Inference

```python
profiler = PerformanceProfiler()

# Profile training
with profiler.profile("training_performance"):
    model.train()
    train_one_epoch(model, train_loader)

# Profile inference
with profiler.profile("inference_performance"):
    model.eval()
    with torch.no_grad():
        evaluate_model(model, test_loader)

# Compare training vs inference efficiency
training_metrics = profiler.get_metrics("training_performance")
inference_metrics = profiler.get_metrics("inference_performance")

print(f"Training throughput: {training_metrics.throughput} samples/s")
print(f"Inference throughput: {inference_metrics.throughput} samples/s")
```

## Troubleshooting

### Common Performance Issues

```python
def diagnose_performance_issues(profiler_results):
    issues = []
    
    # Check for memory issues
    if profiler_results.memory.peak_ram_mb > 16000:  # >16GB
        issues.append("High memory usage - consider reducing batch size or model size")
    
    # Check for CPU bottlenecks
    if profiler_results.system.cpu_percent > 95:
        issues.append("CPU bottleneck - consider data loading optimization or distributed training")
    
    # Check for GPU underutilization
    if profiler_results.compute.gpu_utilization_percent < 70:
        issues.append("GPU underutilized - increase batch size or improve data loading")
    
    # Check for I/O bottlenecks
    if profiler_results.system.disk_io_wait_percent > 20:
        issues.append("I/O bottleneck - optimize data loading or use faster storage")
    
    return issues

# Usage after profiling
metrics = profiler.get_metrics()
issues = diagnose_performance_issues(metrics)
for issue in issues:
    print(f"⚠️ {issue}")
```

---

**Next**: Learn about [Experiment Logging](experiment_logger.md) for comprehensive ML experiment tracking.
