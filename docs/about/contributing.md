# Contributing Guide

Thank you for your interest in contributing to the ML Cookbook! This project aims to provide comprehensive, production-ready measurement tools for the ML community.

## 🎯 Project Vision

The ML Cookbook democratizes professional ML measurement by providing:
- **Comprehensive profiling** tools for performance optimization
- **Statistical validation** for rigorous model comparison
- **Sustainability tracking** for environmentally responsible AI
- **Professional tooling** that scales from individual developers to enterprise teams

## 🤝 How to Contribute

### Types of Contributions Welcome

**🐛 Bug Reports & Fixes**
- Performance profiling edge cases
- Statistical test implementation issues
- Carbon tracking accuracy improvements
- CLI usability problems

**✨ Feature Enhancements**
- New profiling metrics or visualizations
- Additional statistical test methods
- Extended cloud provider support
- Integration with popular ML frameworks

**📚 Documentation Improvements**
- Tutorial expansion and clarification
- API documentation enhancements
- Real-world usage examples
- Multi-language documentation

**🧪 Testing & Quality Assurance**
- Additional unit and integration tests
- Performance benchmarking
- Cross-platform compatibility testing
- Security and reliability improvements

## 🚀 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/ml-cookbook.git
   cd ml-cookbook
   ```

2. **Create Development Environment**
   ```bash
   python -m venv ml-cookbook-dev
   source ml-cookbook-dev/bin/activate  # On Windows: ml-cookbook-dev\\Scripts\\activate
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Installation**
   ```bash
   pytest tests/
   cookbook-prof --help
   ```

### Development Dependencies

The `[dev]` extra includes:
- **Testing**: `pytest`, `pytest-cov`, `pytest-mock`
- **Linting**: `black`, `flake8`, `mypy`
- **Documentation**: `mkdocs-material`, `mkdocstrings`
- **Quality**: `pre-commit`, `bandit`

## 📝 Development Guidelines

### Code Style & Standards

**Python Code Style**
- Follow PEP 8 with Black formatting
- Maximum line length: 88 characters
- Use type hints for all public APIs
- Comprehensive docstrings in Google style

```python
def profile_operation(
    operation_name: str,
    track_gpu: bool = True,
    memory_interval: float = 0.5
) -> PerformanceMetrics:
    """Profile a machine learning operation with comprehensive metrics.
    
    Args:
        operation_name: Descriptive name for the operation being profiled.
        track_gpu: Whether to track GPU memory and utilization.
        memory_interval: Sampling interval for memory tracking in seconds.
        
    Returns:
        PerformanceMetrics object containing comprehensive profiling data.
        
    Raises:
        ProfilerError: If profiling cannot be initialized or fails during execution.
        
    Example:
        >>> profiler = PerformanceProfiler()
        >>> metrics = profiler.profile_operation("model_training", track_gpu=True)
        >>> print(f"Peak memory: {metrics.memory.peak_ram_mb} MB")
    """
```

**Documentation Standards**
- All public APIs must have comprehensive docstrings
- Include practical examples in documentation
- Maintain up-to-date API references
- Write tutorials that are beginner-friendly but technically accurate

### Testing Requirements

**Unit Tests**
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=cookbook tests/

# Run specific test categories
pytest tests/test_profiler.py -v
```

**Integration Tests**
```bash
# Test CLI functionality
pytest tests/integration/

# Test with real ML workloads
pytest tests/integration/test_real_workloads.py
```

**Performance Tests**
```bash
# Benchmark profiler overhead
pytest tests/benchmark/ --benchmark-only
```

### Quality Checks

All contributions must pass:
```bash
# Code formatting
black cookbook/ tests/

# Linting
flake8 cookbook/ tests/

# Type checking
mypy cookbook/

# Security scanning
bandit -r cookbook/
```

## 🏗️ Architecture Overview

Understanding the codebase structure helps in making meaningful contributions:

```
cookbook/
├── measure/              # Main measurement package
│   ├── profiler.py      # Performance profiling core
│   ├── logger.py        # Experiment logging
│   ├── validator.py     # Statistical validation
│   ├── carbon.py        # Carbon tracking
│   └── cli.py           # Command-line interface
├── examples/            # Example notebooks and scripts
├── tests/              # Comprehensive test suite
└── docs/               # Documentation source
```

### Key Design Principles

**Modular Architecture**
- Each component (profiler, logger, validator, carbon) is independently usable
- Clear interfaces between components
- Minimal dependencies between modules

**Performance First**
- Measurement overhead <1% of workload time
- Efficient memory usage even for long-running operations
- Graceful degradation when resources are constrained

**Production Ready**
- Comprehensive error handling and recovery
- Professional logging and debugging support
- Configuration management for different environments

## 🎯 Contribution Areas

### High-Priority Areas

**🔬 Profiler Enhancements**
- Support for distributed training profiling
- Advanced GPU metrics (memory fragmentation, utilization patterns)
- Network I/O profiling for data loading optimization
- Integration with hardware monitoring tools

**📊 Statistical Methods**
- Additional effect size measures (Glass's delta, Hedges' g)
- Non-parametric alternatives for small sample sizes
- Bayesian statistical methods for uncertainty quantification
- Multiple comparison procedures (Bonferroni alternatives)

**🌱 Sustainability Features**
- Integration with carbon offset marketplaces
- Real-time carbon intensity APIs
- Team carbon budgeting and allocation
- Advanced optimization recommendations

**⚙️ Integration & Tooling**
- Kubeflow Pipelines integration
- MLflow tracking integration
- Jupyter notebook extensions
- VS Code extension for inline profiling

### Feature Request Process

1. **Check Existing Issues**: Search for existing feature requests or bugs
2. **Create Detailed Issue**: Use provided templates for consistency
3. **Discussion**: Engage with maintainers on approach and design
4. **Implementation**: Follow coding standards and testing requirements
5. **Review**: Participate in code review process
6. **Documentation**: Update relevant documentation

## 🧪 Testing Guidelines

### Writing Tests

**Unit Test Example**
```python
import pytest
from cookbook.measure import PerformanceProfiler

class TestPerformanceProfiler:
    def test_basic_profiling(self):
        """Test basic profiling functionality."""
        profiler = PerformanceProfiler(track_gpu=False)
        
        with profiler.profile("test_operation"):
            # Simulate some computation
            sum(range(1000))
        
        metrics = profiler.get_metrics()
        assert metrics.timing.wall_time_s > 0
        assert metrics.memory.peak_ram_mb > 0
    
    @pytest.mark.gpu
    def test_gpu_profiling(self):
        """Test GPU profiling if CUDA is available."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        profiler = PerformanceProfiler(track_gpu=True)
        
        with profiler.profile("gpu_operation"):
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x.t())
        
        metrics = profiler.get_metrics()
        assert metrics.memory.peak_gpu_mb > 0
```

**Integration Test Example**
```python
def test_cli_basic_functionality(tmp_path):
    """Test CLI basic profiling command."""
    script_path = tmp_path / "test_script.py"
    script_path.write_text("""
import time
time.sleep(0.1)
print("Test completed")
    """)
    
    result = subprocess.run([
        "cookbook-prof", "profile", 
        "--script", str(script_path),
        "--output", str(tmp_path / "results.json")
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Test completed" in result.stdout
    
    # Check output file was created
    results_file = tmp_path / "results.json"
    assert results_file.exists()
```

### Test Categories

**Mark Tests Appropriately**
```python
@pytest.mark.unit          # Fast unit tests
@pytest.mark.integration   # Integration tests
@pytest.mark.gpu          # GPU-dependent tests  
@pytest.mark.slow         # Long-running tests
@pytest.mark.benchmark    # Performance benchmarks
```

**Run Specific Test Categories**
```bash
# Fast tests only
pytest -m "not slow and not gpu"

# GPU tests
pytest -m gpu

# Benchmarks
pytest -m benchmark --benchmark-only
```

## 📚 Documentation Guidelines

### Writing Documentation

**Tutorial Structure**
- **Overview**: What the feature does and why it's useful
- **Basic Usage**: Simple example that works out of the box
- **Advanced Usage**: Comprehensive examples with configuration options
- **Integration Examples**: How to use with popular ML frameworks
- **Best Practices**: Common patterns and recommendations
- **Troubleshooting**: Common issues and solutions

**API Documentation**
- Comprehensive docstrings for all public methods
- Parameter descriptions with types and defaults
- Return value descriptions
- Example usage for complex methods
- Cross-references to related functionality

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
mkdocs serve

# Build for deployment
mkdocs build
```

## 🚦 Pull Request Process

### Before Submitting

1. **Fork and Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add comprehensive tests
   - Update documentation

3. **Test Everything**
   ```bash
   pytest tests/
   black --check cookbook/ tests/
   flake8 cookbook/ tests/
   mypy cookbook/
   ```

4. **Update Documentation**
   ```bash
   mkdocs build
   ```

### Pull Request Template

When submitting a PR, include:

**Description**
- What does this PR do?
- What problem does it solve?
- Any breaking changes?

**Testing**
- What tests were added?
- How was the change validated?
- Any manual testing performed?

**Documentation**
- What documentation was updated?
- Are there new usage examples?
- Any API changes documented?

**Checklist**
- [ ] Tests pass
- [ ] Code follows style guidelines  
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact assessed

## 🌟 Recognition

### Contributors

All contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Documentation acknowledgments
- Project presentations and talks

### Contribution Types Valued

**Code Contributions**
- Bug fixes and performance improvements
- New features and enhancements
- Test coverage improvements
- Code quality and maintenance

**Non-Code Contributions**
- Documentation improvements
- Tutorial and example creation
- Bug reports and feature requests
- Community support and engagement
- Translation and internationalization

## 📞 Getting Help

### Communication Channels

**GitHub Issues**
- Bug reports: Use bug report template
- Feature requests: Use feature request template
- Questions: Use discussion template

**Development Questions**
- Technical architecture questions
- Implementation approach discussions
- Performance optimization help
- Integration guidance

### Maintainer Response Times

- **Bug reports**: Within 48 hours
- **Feature requests**: Within 1 week  
- **Pull requests**: Within 1 week for initial review
- **Security issues**: Within 24 hours

## 🔒 Security

### Reporting Security Issues

**Do not** create public issues for security vulnerabilities.

**Instead:**
1. Email: security@ml-cookbook-project.org
2. Include detailed description
3. Provide reproduction steps if possible
4. We will respond within 24 hours

### Security Best Practices

- No hardcoded credentials or secrets
- Validate all user inputs
- Use secure defaults for configuration
- Regular dependency security scanning
- Principle of least privilege

## 📄 License

By contributing to ML Cookbook, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to making ML development more measurable, reproducible, and sustainable!** 🚀

Your contributions help democratize professional ML engineering practices and advance the field toward more responsible AI development.
