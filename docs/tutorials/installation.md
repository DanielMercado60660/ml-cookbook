# Installation Guide

## Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Git (for development installation)

## Installation Methods

### 1. Development Installation (Recommended)

For development and customization:

```bash
# Clone the repository
git clone https://github.com/yourusername/ml-cookbook.git
cd ml-cookbook

# Create virtual environment (recommended)
python -m venv ml-cookbook-env
source ml-cookbook-env/bin/activate  # On Windows: ml-cookbook-env\Scripts\activate

# Install in development mode
pip install -e .

# Verify installation
cookbook-prof --help
```

### 2. PyPI Installation (Coming Soon)

Once published to PyPI:

```bash
pip install ml-cookbook
```

## Optional Dependencies

### For GPU Support

```bash
# PyTorch with CUDA (check pytorch.org for your specific CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### For Advanced Carbon Tracking

```bash
# CodeCarbon for real carbon footprint measurement
pip install codecarbon

# Additional cloud provider support
pip install boto3  # For AWS
pip install google-cloud-monitoring  # For GCP
```

### For Advanced Logging Backends

```bash
# Weights & Biases
pip install wandb

# TensorBoard
pip install tensorboard

# MLflow
pip install mlflow
```

## Verification

Test your installation:

```python
from cookbook.measure import (
    PerformanceProfiler,
    ExperimentLogger,
    StatisticalValidator, 
    CarbonTracker
)

print("✅ ML Cookbook installed successfully!")

# Quick test
profiler = PerformanceProfiler()
print(f"✅ Performance profiler ready")

logger = ExperimentLogger()
print(f"✅ Experiment logger ready")

validator = StatisticalValidator()
print(f"✅ Statistical validator ready")

tracker = CarbonTracker()
print(f"✅ Carbon tracker ready")
```

## CLI Verification

```bash
# Test CLI installation
cookbook-prof --version

# View available commands
cookbook-prof --help

# Test basic profiling
cookbook-prof profile --help
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# If you see import errors, ensure you're in the right environment
which python
pip list | grep cookbook

# Reinstall in development mode
pip install -e . --force-reinstall
```

#### CLI Not Found

```bash
# If cookbook-prof command not found
pip install -e . --force-reinstall

# Check if it's in your PATH
which cookbook-prof

# Or run directly
python -m cookbook.measure.cli --help
```

#### GPU/CUDA Issues

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CodeCarbon Issues

```bash
# If carbon tracking fails
pip install codecarbon --upgrade

# Test CodeCarbon directly
python -c "from codecarbon import EmissionsTracker; print('CodeCarbon working')"
```

### Development Setup

For contributing to the project:

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/ml-cookbook.git
cd ml-cookbook

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
black cookbook/
flake8 cookbook/
```

## Next Steps

Once installed, proceed to the [Quick Start Guide](quick_start.md) to begin using the ML Cookbook.

## Support

- 📚 [Documentation](../index.md)
- 🐛 [Report Issues](https://github.com/yourusername/ml-cookbook/issues)
- 💬 [Discussions](https://github.com/yourusername/ml-cookbook/discussions)
