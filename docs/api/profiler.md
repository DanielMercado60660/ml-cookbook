# Profiler API Reference

## PerformanceProfiler

::: cookbook.measure.profiler.PerformanceProfiler

### Overview

The PerformanceProfiler is the core class for tracking system resources, timing, and compute metrics during ML workloads.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `track_gpu` | bool | `True` | Enable GPU memory and utilization tracking |
| `track_carbon` | bool | `False` | Enable carbon footprint measurement |
| `detailed_memory` | bool | `False` | Enable detailed memory profiling with leak detection |
| `memory_interval` | float | `0.5` | Memory sampling interval in seconds |
| `enable_line_profiling` | bool | `False` | Enable line-by-line performance profiling |
| `estimate_flops` | bool | `False` | Enable FLOPS estimation for operations |

### Main Methods

#### profile(operation_name: str)
Context manager for profiling a specific operation.

```python
profiler = PerformanceProfiler()
with profiler.profile("training_loop"):
    model.fit(X_train, y_train)
```

**Parameters:**
- `operation_name` (str): Descriptive name for the operation being profiled

**Returns:**
- Context manager that handles profiling lifecycle

#### get_metrics(operation_name: Optional[str] = None) → PerformanceMetrics
Retrieve comprehensive performance metrics.

```python
metrics = profiler.get_metrics()
print(f"Peak RAM: {metrics.memory.peak_ram_mb} MB")
```

**Parameters:**
- `operation_name` (str, optional): Specific operation to get metrics for. If None, returns metrics for the last operation.

**Returns:**
- `PerformanceMetrics`: Comprehensive performance data

#### get_current_memory() → float
Get current memory usage in MB.

```python
current_ram = profiler.get_current_memory()
```

**Returns:**
- float: Current RAM usage in megabytes

#### get_timing_breakdown() → Dict[str, float]
Get timing breakdown for all profiled operations.

```python
breakdown = profiler.get_timing_breakdown()
for operation, time_s in breakdown.items():
    print(f"{operation}: {time_s:.3f}s")
```

**Returns:**
- Dict[str, float]: Operation names mapped to execution times in seconds

## PerformanceMetrics

Data class containing comprehensive performance measurement results.

### Attributes

#### memory: MemoryMetrics
Memory usage information including:
- `peak_ram_mb`: Peak RAM usage in megabytes
- `final_ram_mb`: Final RAM usage in megabytes  
- `peak_gpu_mb`: Peak GPU memory usage in megabytes (if GPU tracking enabled)
- `memory_leak_detected`: Boolean indicating potential memory leaks

#### timing: TimingMetrics
Timing information including:
- `wall_time_s`: Wall clock execution time in seconds
- `cpu_time_s`: CPU execution time in seconds
- `user_time_s`: User CPU time in seconds
- `system_time_s`: System CPU time in seconds

#### compute: ComputeMetrics
Computational metrics including:
- `estimated_flops`: Estimated floating point operations (if enabled)
- `flops_per_second`: FLOPS per second throughput
- `gpu_utilization_percent`: Average GPU utilization percentage
- `cpu_utilization_percent`: Average CPU utilization percentage

#### system: SystemMetrics
System-level metrics including:
- `cpu_percent`: CPU usage percentage
- `memory_percent`: Memory usage percentage
- `disk_io_percent`: Disk I/O usage percentage
- `network_io_mb`: Network I/O in megabytes

### Methods

#### to_dict() → Dict[str, Any]
Convert metrics to dictionary format for serialization.

```python
metrics_dict = metrics.to_dict()
```

#### summary() → str
Generate human-readable summary string.

```python
print(metrics.summary())
# Output: "Peak RAM: 2.4 GB, Wall Time: 45.2s, GPU Util: 78%"
```

## Usage Examples

### Basic Performance Profiling

```python
from cookbook.measure import PerformanceProfiler

profiler = PerformanceProfiler(track_gpu=True)

with profiler.profile("model_training"):
    # Your ML training code
    for epoch in range(10):
        train_epoch()

metrics = profiler.get_metrics()
print(f"Training completed in {metrics.timing.wall_time_s:.1f}s")
print(f"Peak memory usage: {metrics.memory.peak_ram_mb:.1f} MB")
```

### GPU Memory Monitoring

```python
profiler = PerformanceProfiler(track_gpu=True, detailed_memory=True)

with profiler.profile("gpu_intensive_task"):
    model = LargeModel().cuda()
    
    for batch in dataloader:
        output = model(batch.cuda())
        
        # Check GPU memory during training
        if profiler.get_current_gpu_memory() > 0.9:  # >90% usage
            print("Warning: High GPU memory usage")

gpu_metrics = profiler.get_metrics()
print(f"Peak GPU memory: {gpu_metrics.memory.peak_gpu_mb} MB")
```

### Multi-Operation Profiling

```python
profiler = PerformanceProfiler()

# Profile data loading
with profiler.profile("data_loading"):
    data = load_dataset()

# Profile preprocessing  
with profiler.profile("preprocessing"):
    processed_data = preprocess(data)

# Profile training
with profiler.profile("training"):
    model = train_model(processed_data)

# Get breakdown of all operations
timing_breakdown = profiler.get_timing_breakdown()
total_time = sum(timing_breakdown.values())

print("Operation breakdown:")
for op, time_s in timing_breakdown.items():
    percentage = (time_s / total_time) * 100
    print(f"  {op}: {time_s:.2f}s ({percentage:.1f}%)")
```

### Memory Leak Detection

```python
profiler = PerformanceProfiler(detailed_memory=True, memory_interval=0.1)

with profiler.profile("memory_leak_check"):
    for i in range(100):
        # Code that might leak memory
        data = allocate_data()
        process_data(data)
        # Forgot to clean up data!

metrics = profiler.get_metrics()
if metrics.memory.memory_leak_detected:
    print("⚠️ Potential memory leak detected")
    print(f"Memory grew from {metrics.memory.initial_ram_mb}MB to {metrics.memory.final_ram_mb}MB")
```

### FLOPS Estimation

```python
import torch
import torch.nn as nn

profiler = PerformanceProfiler(estimate_flops=True)

model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

with profiler.profile("flops_analysis"):
    x = torch.randn(32, 784)
    output = model(x)

compute_metrics = profiler.get_metrics().compute
print(f"Estimated FLOPs: {compute_metrics.estimated_flops:,}")
print(f"FLOPS per second: {compute_metrics.flops_per_second:,.0f}")
```

## Error Handling

The PerformanceProfiler handles errors gracefully and provides partial results even if profiling is interrupted:

```python
profiler = PerformanceProfiler()

try:
    with profiler.profile("potentially_failing_operation"):
        risky_operation()  # This might raise an exception
except Exception as e:
    print(f"Operation failed: {e}")

# Still get partial profiling results
if profiler.has_results():
    partial_metrics = profiler.get_metrics()
    print(f"Partial timing: {partial_metrics.timing.wall_time_s:.2f}s")
```

## Best Practices

### 1. Use appropriate sampling intervals
```python
# For long-running operations, use lower frequency sampling
profiler = PerformanceProfiler(memory_interval=1.0)  # Sample every second

# For detailed analysis, use higher frequency
profiler = PerformanceProfiler(memory_interval=0.1)  # Sample every 100ms
```

### 2. Enable GPU tracking only when needed
```python
import torch

# Conditional GPU tracking based on availability
profiler = PerformanceProfiler(track_gpu=torch.cuda.is_available())
```

### 3. Use descriptive operation names
```python
# Good: Descriptive names
with profiler.profile("resnet50_forward_pass"):
    output = model(input_batch)

# Bad: Generic names  
with profiler.profile("operation"):
    output = model(input_batch)
```

### 4. Profile representative workloads
```python
# Profile with realistic data sizes
with profiler.profile("production_inference"):
    # Use production-like batch sizes
    batch = torch.randn(128, 3, 224, 224)  # Realistic batch
    predictions = model(batch)
```
