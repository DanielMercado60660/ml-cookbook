# Quick Start Guide

Get up and running with the ML Cookbook measurement suite in under 5 minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-cookbook.git
cd ml-cookbook

# Install in development mode
pip install -e .

# Verify installation
cookbook-prof --help
```

## Basic Usage

### 1. 🔬 Performance Profiling

```python
from cookbook.measure import PerformanceProfiler

profiler = PerformanceProfiler(track_gpu=True, track_carbon=True)

with profiler.profile("training_loop"):
    # Your ML code here
    model.fit(X_train, y_train)
    
# Get comprehensive metrics
metrics = profiler.get_metrics()
print(f"Peak RAM: {metrics.memory.peak_ram_mb:.1f} MB")
```

### 2. 📊 Experiment Logging

```python
from cookbook.measure import ExperimentLogger, ExperimentConfig

config = ExperimentConfig(
    project_name="my-project",
    experiment_name="baseline-model",
    hyperparameters={"lr": 0.001, "batch_size": 32}
)

with ExperimentLogger(config) as logger:
    for epoch in range(10):
        loss = train_epoch()
        logger.log_metrics({"train_loss": loss}, step=epoch)
```

### 3. 📈 Statistical Validation

```python
from cookbook.measure import StatisticalValidator, TestType

validator = StatisticalValidator()

# Compare two models
result = validator.compare_models(
    baseline_scores=[0.87, 0.89, 0.86],
    variant_scores=[0.91, 0.93, 0.90],
    test_type=TestType.WELCH_T_TEST
)

print(f"Significant improvement: {result.is_significant}")
print(f"Effect size: {result.effect_size.interpretation}")
```

### 4. 🌱 Carbon Tracking

```python
from cookbook.measure import CarbonTracker

tracker = CarbonTracker(
    project_name="sustainable-ml",
    cloud_provider="gcp"
)

with tracker.start_tracking("model_training"):
    # Your energy-intensive ML code
    model.train()

metrics = tracker.stop_tracking()
print(f"Emissions: {metrics.emissions_kg_co2 * 1000:.2f}g CO2eq")
```

### 5. 🚀 Complete Integration

```python
from cookbook.measure import CarbonAwareProfiler

# Combined profiling (performance + carbon)
profiler = CarbonAwareProfiler(
    project_name="my-project",
    track_carbon=True
)

with profiler.profile_with_carbon("complete_training") as session:
    # Full ML pipeline
    model = create_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

# Results include both performance and carbon metrics!
```

## CLI Usage

```bash
# Launch experiment with configuration
cookbook-prof run --config experiments/config.yaml

# Quick profiling of any Python script  
cookbook-prof profile --script train_model.py --track-carbon

# Generate comprehensive reports
cookbook-prof report --experiment-dir ./logs --output report.html
```

## Next Steps

- 📖 Read the [detailed tutorials](performance_profiler.md) for each component
- 🧪 Try the [example notebooks](../examples/performance_profiling.md)
- 🔧 Check out the [API reference](../api/profiler.md) for advanced usage
- 🚀 See the [complete pipeline demo](../examples/complete_pipeline.md) for a full workflow

## Getting Help

- 📚 Check the [troubleshooting guide](../advanced/troubleshooting.md)
- 🐛 Report issues on [GitHub](https://github.com/yourusername/ml-cookbook/issues)
- 💬 Join the discussion in [GitHub Discussions](https://github.com/yourusername/ml-cookbook/discussions)

---

**Next**: Learn about [Performance Profiling](performance_profiler.md) in detail.
