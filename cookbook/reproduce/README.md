# Reproducibility Toolkit

**Project 1.2: The Reproducibility Toolkit** - Create tools and standards to ensure experiments are reliably reproducible.

This module provides a comprehensive suite of utilities to ensure that ML experiments can be reliably reproduced across different environments, hardware configurations, and time periods.

## 🎯 Key Features

- **Global Seed Management**: Centralized seeding across all ML libraries (PyTorch, JAX, NumPy, etc.)
- **Deterministic Operations**: Framework-specific configurations for deterministic behavior
- **Checkpoint Verification**: Cryptographic verification of model checkpoints and artifacts
- **Project Validation**: Tools to validate projects follow reproducibility best practices

## 🚀 Quick Start

```python
from cookbook.reproduce import set_global_seed, enable_deterministic_mode

# Set up reproducible environment
set_global_seed(42)
enable_deterministic_mode()

# Your experiment code here...
# All random operations will now be deterministic
```

## 📚 Core Components

### 1. Global Seeding (`seeding.py`)

Manages seeds across all major ML libraries with context management support:

```python
from cookbook.reproduce import set_global_seed, SeedContext, verify_deterministic_execution

# Set global seed for entire experiment
set_global_seed(42, deterministic=True)

# Temporarily change seed for specific operation
with SeedContext(123):
    # Operations with seed 123
    data = np.random.rand(10)
# Back to seed 42

# Verify function produces deterministic outputs
is_deterministic = verify_deterministic_execution(my_model_forward, input_tensor)
```

**Key Functions:**
- `set_global_seed(seed, deterministic=True)` - Set global seed across all libraries
- `SeedContext(seed)` - Context manager for temporary seed changes
- `verify_deterministic_execution(func, *args, **kwargs)` - Test function determinism
- `generate_deterministic_seeds(base_seed, count)` - Generate seed sequences for distributed training
- `save_seed_state(filepath)` / `load_seed_state(filepath)` - Persist/restore RNG states

### 2. Deterministic Operations (`deterministic.py`)

Configures ML frameworks for deterministic behavior:

```python
from cookbook.reproduce import enable_deterministic_mode, DeterministicConfig, DeterministicContext

# Simple deterministic mode
previous_state = enable_deterministic_mode()

# Advanced configuration
config = DeterministicConfig(
    torch_deterministic=True,
    torch_benchmark=False,  # Sacrifice speed for determinism
    warn_performance=True
)
config.apply()

# Context manager for temporary deterministic mode
with DeterministicContext(config):
    # Deterministic operations
    output = model(input_data)
# Back to original settings
```

**Key Classes:**
- `DeterministicConfig` - Configuration for deterministic behavior
- `DeterministicContext` - Context manager for temporary deterministic mode

**Key Functions:**
- `enable_deterministic_mode()` - Enable deterministic mode across all frameworks
- `get_deterministic_status()` - Check current deterministic settings
- `benchmark_deterministic_impact()` - Measure performance impact of deterministic mode

### 3. Checkpoint Verification (`verification.py`)

Ensures integrity of model checkpoints and experimental artifacts:

```python
from cookbook.reproduce import CheckpointVerifier, compute_checkpoint_hash

# Simple hash computation
hash_value = compute_checkpoint_hash("model.pth")
print(f"Checkpoint hash: {hash_value}")

# Advanced verification with metadata
verifier = CheckpointVerifier("experiments/")
verifier.register_checkpoint("model.pth", metadata={"epoch": 10, "accuracy": 0.95})

# Verify integrity
is_valid = verifier.verify_checkpoint("model.pth")

# Verify all checkpoints
results = verifier.verify_all_checkpoints()
```

**Key Classes:**
- `CheckpointVerifier` - Manages checkpoint verification for experiment directories
- `CheckpointMetadata` - Metadata container for checkpoint information

**Key Functions:**
- `compute_checkpoint_hash(path, algorithm="sha256")` - Compute cryptographic hash
- `verify_checkpoint_integrity(path, expected_hash)` - Verify against expected hash
- `create_integrity_manifest(directory)` - Create manifest for entire directory

### 4. Project Validation (`validation.py`)

Validate reproducible project structures:

```python
from cookbook.reproduce import validate_template_structure

# Validate existing project follows reproducibility best practices
validation_results = validate_template_structure("existing_project/")
print(f"Validation score: {validation_results['score']:.2f}")

if validation_results['errors']:
    print("Errors found:")
    for error in validation_results['errors']:
        print(f"  - {error}")

if validation_results['warnings']:
    print("Warnings:")
    for warning in validation_results['warnings']:
        print(f"  - {warning}")
```

**Validation Checks:**
- Required files (requirements.txt, pyproject.toml, README.md, .gitignore)
- Required directories (src/, tests/)
- Configuration management (config.yaml or similar)
- Seed management in code
- Deterministic configuration
- Reproducibility test coverage

## 🔬 Complete Reproducibility Workflow

Here's a complete example showing all components working together:

```python
import tempfile
from pathlib import Path
from cookbook.reproduce import (
    set_global_seed, enable_deterministic_mode, 
    CheckpointVerifier, create_reproducible_template,
    TemplateConfig, validate_template_structure
)

# 1. Create a reproducible project
config = TemplateConfig(
    project_name="demo_experiment",
    frameworks=["pytorch"]
)
create_reproducible_template("demo_experiment", config)

# 2. Set up reproducible environment
set_global_seed(42)
enable_deterministic_mode()

# 3. Initialize checkpoint verification
verifier = CheckpointVerifier("demo_experiment/checkpoints/")

# 4. Run experiment with checkpointing
import torch
model = torch.nn.Linear(10, 1)

# Save with verification
torch.save(model.state_dict(), "demo_experiment/checkpoints/model.pth")
hash_value = verifier.register_checkpoint(
    "model.pth", 
    metadata={"parameters": sum(p.numel() for p in model.parameters())}
)

# 5. Verify integrity
is_valid = verifier.verify_checkpoint("model.pth")
print(f"Checkpoint valid: {is_valid}")

# 6. Generate verification report
verifier.export_verification_report("demo_experiment/verification_report.md")

# 7. Validate project structure
validation = validate_template_structure("demo_experiment/")
print(f"Project validation score: {validation['score']:.2f}")
```

## ⚡ Performance Considerations

Deterministic mode can impact performance. Use the benchmarking tools to measure the impact:

```python
from cookbook.reproduce import benchmark_deterministic_impact

def my_training_step():
    # Your training step here
    return model(batch)

# Measure performance impact
results = benchmark_deterministic_impact(my_training_step, runs=5)
print(f"Slowdown factor: {results['slowdown_factor']:.2f}x")
print(f"Absolute difference: {results['absolute_difference']:.3f}s")
```

## 🧪 Testing Reproducibility

The toolkit includes utilities to verify that your experiments are truly reproducible:

```python
from cookbook.reproduce import verify_deterministic_execution

def run_experiment():
    # Your experiment logic
    data = torch.randn(100, 10)
    output = model(data)
    return output.sum().item()

# Verify experiment is deterministic
set_global_seed(42)
is_deterministic = verify_deterministic_execution(run_experiment)
assert is_deterministic, "Experiment is not deterministic!"
```

## ✅ Quality Gates

To meet **Project 1.2 Quality Gates**, ensure your experiments pass these tests:

```python
# Test 1: Identical metrics across 3 consecutive runs
def test_reproducibility():
    results = []
    for i in range(3):
        set_global_seed(42)
        enable_deterministic_mode()
        result = run_experiment()
        results.append(result)
    
    # All results should be identical (within tolerance)
    for i in range(1, len(results)):
        assert abs(results[0] - results[i]) < 1e-10

# Test 2: Checkpoint verification passes
def test_checkpoint_integrity():
    verifier = CheckpointVerifier("experiments/")
    results = verifier.verify_all_checkpoints()
    assert all(results.values()), "Some checkpoints failed verification"

# Test 3: Project structure validation
def test_project_structure():
    validation = validate_template_structure(".")
    assert validation['valid'], f"Project validation failed: {validation['errors']}"
```

## 🐳 CI/CD Integration

Add this to your GitHub Actions workflow for automated reproducibility testing:

```yaml
name: Reproducibility Tests
on: [push, pull_request]

jobs:
  test-reproducibility:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Test reproducibility
      run: |
        python -m pytest tests/test_reproducibility.py -v
        
    - name: Validate project structure
      run: |
        python -c "
        from cookbook.reproduce import validate_template_structure
        result = validate_template_structure('.')
        assert result['valid'], f'Validation failed: {result[\"errors\"]}'
        print(f'Validation score: {result[\"score\"]:.2f}')
        "
```

## 🔧 Advanced Configuration

### Custom Hash Algorithms

```python
# Use different hash algorithms for different purposes
sha256_hash = compute_checkpoint_hash("model.pth", algorithm="sha256")  # Standard
md5_hash = compute_checkpoint_hash("model.pth", algorithm="md5")        # Fast
sha512_hash = compute_checkpoint_hash("model.pth", algorithm="sha512")  # Secure
```

### Distributed Training Support

```python
# Generate deterministic seeds for distributed training
base_seed = 42
worker_seeds = generate_deterministic_seeds(base_seed, num_workers=4)

# Each worker uses its assigned seed
worker_id = int(os.environ.get('RANK', 0))
set_global_seed(worker_seeds[worker_id])
```

### Custom Verification Metadata

```python
verifier = CheckpointVerifier("experiments/")

# Rich metadata for better tracking
metadata = {
    "experiment_id": "exp_001",
    "model_architecture": "transformer",
    "num_parameters": 175_000_000,
    "training_data": "common_crawl_v1",
    "performance_metrics": {
        "accuracy": 0.95,
        "f1_score": 0.93,
        "training_time_hours": 24.5
    },
    "hardware": {
        "gpu_type": "A100",
        "num_gpus": 8,
        "memory_gb": 640
    }
}

verifier.register_checkpoint("final_model.pth", metadata=metadata)
```

## 🚨 Common Issues and Solutions

### Issue 1: Non-deterministic behavior despite seeding

**Cause**: Some operations are inherently non-deterministic (e.g., some CUDA operations)

**Solution**:
```python
# Enable strict deterministic mode (may impact performance)
config = DeterministicConfig(
    torch_deterministic=True,
    torch_use_deterministic_algorithms=True,
    torch_warn_only=False  # Fail on non-deterministic operations
)
config.apply()
```

### Issue 2: Performance degradation in deterministic mode

**Cause**: Deterministic algorithms are often slower than optimized non-deterministic ones

**Solution**:
```python
# Benchmark and decide trade-offs
impact = benchmark_deterministic_impact(training_step)
if impact['slowdown_factor'] > 2.0:  # More than 2x slower
    logger.warning("Consider disabling deterministic mode for production training")
```

### Issue 3: Checkpoint verification fails after model updates

**Cause**: Model architecture changed, invalidating old checkpoints

**Solution**:
```python
# Version your checkpoints
metadata = {
    "model_version": "v2.1",
    "architecture_hash": hash(str(model)),
    "backward_compatible": False
}
verifier.register_checkpoint("model_v2.pth", metadata=metadata)
```

## 📖 References

- [PyTorch Reproducibility Documentation](https://pytorch.org/docs/stable/notes/randomness.html)
- [JAX Deterministic Compilation](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#random-numbers)
- [ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code)

## 🤝 Contributing

When contributing to the reproducibility toolkit:

1. All new features must include reproducibility tests
2. Performance impact should be documented and benchmarked
3. Template changes must be validated against existing projects
4. Documentation should include complete examples

---

**Next**: Proceed to **Project 1.3: The "Curated Corpus" Project** to build the data processing foundation.
