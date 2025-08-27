# Basic Usage Guide

This guide covers the fundamental usage patterns for the ML Cookbook measurement suite.

## Core Components Overview

The ML Cookbook provides four main measurement components:

| Component | Purpose | Key Use Cases |
|-----------|---------|---------------|
| 🔬 **PerformanceProfiler** | Resource tracking | Memory, timing, FLOPS analysis |
| 📊 **ExperimentLogger** | Experiment management | Hyperparameter tracking, results logging |
| 📈 **StatisticalValidator** | Rigorous testing | A/B testing, significance analysis |
| 🌱 **CarbonTracker** | Sustainability | Environmental impact measurement |

## Basic Usage Patterns

### 1. Context Manager Pattern (Recommended)

```python
from cookbook.measure import PerformanceProfiler

profiler = PerformanceProfiler()

with profiler.profile("training_loop"):
    # Your ML code here
    model.fit(X_train, y_train)

# Automatically get results
metrics = profiler.get_metrics()
```

### 2. Explicit Start/Stop Pattern

```python
from cookbook.measure import CarbonTracker

tracker = CarbonTracker(project_name="my-project")

tracker.start_tracking("data_preprocessing")
# Your preprocessing code
data = preprocess_data(raw_data)
metrics = tracker.stop_tracking("data_preprocessing")
```

### 3. Integrated Profiling

```python
from cookbook.measure import CarbonAwareProfiler

# Combined performance + carbon tracking
profiler = CarbonAwareProfiler(
    project_name="sustainable-ml",
    track_carbon=True
)

with profiler.profile_with_carbon("complete_pipeline") as session:
    # Full ML pipeline
    data = preprocess(raw_data)
    model = train_model(data)
    results = evaluate_model(model, test_data)

# Get both performance and carbon metrics
combined_results = session.results
```

## Configuration Patterns

### Experiment Configuration

```python
from cookbook.measure import ExperimentLogger, ExperimentConfig

config = ExperimentConfig(
    project_name="neural-architecture-search",
    experiment_name="resnet-baseline",
    tags=["computer-vision", "baseline"],
    hyperparameters={
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 50
    },
    metadata={
        "author": "ML Team",
        "environment": "development"
    }
)

logger = ExperimentLogger(config)
logger.start_experiment()

# Log metrics during training
for epoch in range(config.hyperparameters["epochs"]):
    loss = train_epoch()
    accuracy = validate()
    
    logger.log_metrics({
        "train_loss": loss,
        "val_accuracy": accuracy
    }, step=epoch)

logger.end_experiment()
```

### Statistical Validation

```python
from cookbook.measure import StatisticalValidator, TestType

validator = StatisticalValidator()

# Compare two model variants
baseline_scores = [0.87, 0.89, 0.86, 0.88, 0.85]
new_model_scores = [0.91, 0.93, 0.90, 0.92, 0.89]

result = validator.compare_models(
    baseline_scores=baseline_scores,
    variant_scores=new_model_scores,
    test_type=TestType.WELCH_T_TEST,
    alpha=0.05
)

print(f"Significant improvement: {result.is_significant}")
print(f"Effect size: {result.effect_size.interpretation}")
```

## Error Handling

Always use proper error handling with measurement tools:

```python
from cookbook.measure import PerformanceProfiler

profiler = PerformanceProfiler()

try:
    with profiler.profile("risky_operation"):
        # Code that might fail
        risky_ml_operation()
except Exception as e:
    print(f"Operation failed: {e}")
    # Profiler automatically handles cleanup
finally:
    # Always get whatever metrics were collected
    if profiler.has_results():
        metrics = profiler.get_metrics()
        print(f"Partial results: {metrics.summary()}")
```

## CLI Usage

Use the command-line interface for quick profiling:

```bash
# Profile any Python script
cookbook-prof profile --script train_model.py --track-carbon

# Run with configuration file
cookbook-prof run --config experiments/baseline.yaml

# Generate reports from logs
cookbook-prof report --experiment-dir ./logs --output report.html
```

## Best Practices

### 1. Consistent Naming

```python
# Use consistent, descriptive names
with profiler.profile("data_preprocessing_phase1"):
    clean_data = preprocess_step1(raw_data)

with profiler.profile("model_training_resnet18"):
    model = train_resnet18(clean_data)

with profiler.profile("model_evaluation_final"):
    results = evaluate_model(model, test_data)
```

### 2. Hierarchical Organization

```python
# Organize measurements hierarchically
with profiler.profile("complete_pipeline"):
    with profiler.profile("data_loading"):
        data = load_dataset()
    
    with profiler.profile("feature_engineering"):
        features = engineer_features(data)
    
    with profiler.profile("model_training"):
        model = train_model(features)
```

### 3. Resource-Aware Configuration

```python
# Configure based on available resources
import torch

profiler = PerformanceProfiler(
    track_gpu=torch.cuda.is_available(),
    track_carbon=True,
    detailed_memory=True if torch.cuda.is_available() else False
)
```

## Common Workflows

### Research Workflow

```python
# Research: Compare multiple approaches
approaches = ["random_forest", "neural_network", "gradient_boosting"]
results = {}

for approach in approaches:
    with profiler.profile(f"{approach}_training"):
        model = train_model(approach, data)
        accuracy = evaluate_model(model, test_data)
        results[approach] = accuracy

# Statistical validation
validator = StatisticalValidator()
for i, approach1 in enumerate(approaches):
    for approach2 in approaches[i+1:]:
        comparison = validator.compare_models(
            results[approach1], results[approach2]
        )
        print(f"{approach1} vs {approach2}: {comparison.is_significant}")
```

### Production Monitoring

```python
# Production: Monitor model serving performance
from datetime import datetime

profiler = PerformanceProfiler()
carbon_tracker = CarbonTracker(project_name="production-serving")

# Monitor batch inference
batch_size = 1000
with profiler.profile(f"batch_inference_{datetime.now().isoformat()}"):
    with carbon_tracker.start_tracking("inference_batch"):
        predictions = model.predict(batch_data)
        
        # Check performance thresholds
        metrics = profiler.get_metrics()
        if metrics.timing.wall_time_s > SLA_THRESHOLD:
            alert_on_call_team("Performance SLA violation")
```

### Team Development

```python
# Team: Consistent experiment tracking
config = ExperimentConfig(
    project_name="team-model-optimization",
    experiment_name=f"developer_{os.getenv('USER')}_experiment_{datetime.now().strftime('%Y%m%d_%H%M')}",
    tags=["team-shared", "optimization", f"developer-{os.getenv('USER')}"],
    hyperparameters=load_hyperparameters_from_config(),
    metadata={
        "developer": os.getenv('USER'),
        "branch": get_git_branch(),
        "commit": get_git_commit(),
        "environment": "development"
    }
)

logger = ExperimentLogger(config)
# ... rest of experiment
```

## Integration Examples

### With PyTorch Lightning

```python
import pytorch_lightning as pl
from cookbook.measure import PerformanceProfiler, ExperimentLogger

class LightningModelWithProfiling(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.profiler = PerformanceProfiler()
        self.logger_cookbook = ExperimentLogger(config)
        
    def training_step(self, batch, batch_idx):
        with self.profiler.profile(f"training_step_{batch_idx}"):
            # Your training logic
            loss = self.compute_loss(batch)
            
            # Log to both Lightning and ML Cookbook
            self.log('train_loss', loss)
            self.logger_cookbook.log_metrics({'train_loss': loss}, step=batch_idx)
            
            return loss
```

### With Hugging Face Transformers

```python
from transformers import Trainer
from cookbook.measure import CarbonAwareProfiler

class SustainableTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.carbon_profiler = CarbonAwareProfiler()
    
    def train(self):
        with self.carbon_profiler.profile_with_carbon("huggingface_training"):
            result = super().train()
            
            # Log carbon metrics
            carbon_results = self.carbon_profiler.get_last_results()
            self.log_metrics(carbon_results['carbon'])
            
            return result
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed with `pip install -e .`
2. **GPU Tracking**: Requires PyTorch with CUDA support
3. **Carbon Tracking**: Install CodeCarbon with `pip install codecarbon`
4. **Permissions**: Ensure write access to log directories

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

profiler = PerformanceProfiler(debug=True)
# Will show detailed profiling information
```

---

**Next**: Learn about specific components in the detailed tutorials.
