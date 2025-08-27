# ML Cookbook - Automated Setup Script
# Run this to create all necessary __init__.py files and setup.py

import os
from pathlib import Path


def create_cookbook_init():
    """Create cookbook/__init__.py"""
    content = '''"""
ML Cookbook - Professional ML Engineering Toolkit

A comprehensive suite of tools for ML experiment measurement, logging, 
statistical validation, and sustainability tracking.

Components:
- Performance Profiler: Memory, timing, and compute metrics
- Experiment Logger: Multi-backend experiment tracking  
- Statistical Validator: Rigorous A/B testing framework
- Carbon Tracker: Sustainability and carbon footprint tracking
- CLI Interface: Professional command-line toolkit

Author: Daniel Mercado
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Daniel Mercado"
__email__ = "daniel.mercado@example.com"  # Update this
__description__ = "Professional ML Engineering Toolkit"

# Make key components easily accessible
try:
    from cookbook.measure import (
        PerformanceProfiler,
        ExperimentLogger, 
        ExperimentConfig,
        StatisticalValidator,
        TestType,
        EffectSize,
        SimpleCarbonTracker,
        CookbookCLI
    )

    __all__ = [
        "PerformanceProfiler",
        "ExperimentLogger",
        "ExperimentConfig", 
        "StatisticalValidator",
        "TestType",
        "EffectSize",
        "SimpleCarbonTracker",
        "CookbookCLI",
        "__version__",
        "__author__",
        "__description__"
    ]
except ImportError as e:
    print(f"Warning: Some components not available - {e}")
    __all__ = ["__version__", "__author__", "__description__"]
'''

    with open("cookbook/__init__.py", "w") as f:
        f.write(content)
    print("✅ Created cookbook/__init__.py")


def create_measure_init():
    """Create cookbook/measure/__init__.py"""
    content = '''"""
Measurement Suite - Core measurement and validation tools

This module contains all the essential components for ML experiment
measurement, logging, statistical validation, and carbon tracking.
"""

# Import main components with error handling
__all__ = []

try:
    from .profiler import PerformanceProfiler
    __all__.append("PerformanceProfiler")
    PROFILER_AVAILABLE = True
except ImportError as e:
    PROFILER_AVAILABLE = False
    print(f"Warning: PerformanceProfiler not available - {e}")

try:
    from .logger import ExperimentLogger, ExperimentConfig
    __all__.extend(["ExperimentLogger", "ExperimentConfig"])
    LOGGER_AVAILABLE = True
except ImportError as e:
    LOGGER_AVAILABLE = False
    print(f"Warning: ExperimentLogger not available - {e}")

try:
    from .validator import StatisticalValidator, TestType, EffectSize, StatisticalResult
    __all__.extend(["StatisticalValidator", "TestType", "EffectSize", "StatisticalResult"])
    VALIDATOR_AVAILABLE = True
except ImportError as e:
    VALIDATOR_AVAILABLE = False
    print(f"Warning: StatisticalValidator not available - {e}")

try:
    from .carbon import SimpleCarbonTracker, CarbonMetrics
    __all__.extend(["SimpleCarbonTracker", "CarbonMetrics"])
    CARBON_AVAILABLE = True
except ImportError as e:
    CARBON_AVAILABLE = False
    print(f"Warning: SimpleCarbonTracker not available - {e}")

try:
    from .cli import CookbookCLI
    __all__.append("CookbookCLI")
    CLI_AVAILABLE = True
except ImportError as e:
    CLI_AVAILABLE = False
    print(f"Warning: CookbookCLI not available - {e}")

# Component status
COMPONENT_STATUS = {
    "profiler": PROFILER_AVAILABLE,
    "logger": LOGGER_AVAILABLE,
    "validator": VALIDATOR_AVAILABLE, 
    "carbon": CARBON_AVAILABLE,
    "cli": CLI_AVAILABLE
}

def get_available_components():
    """Return list of available components"""
    return [name for name, available in COMPONENT_STATUS.items() if available]

def print_status():
    """Print status of all components"""
    print("ML Cookbook - Component Status:")
    for component, available in COMPONENT_STATUS.items():
        status = "✅" if available else "❌"
        print(f"  {status} {component}")
'''

    with open("cookbook/measure/__init__.py", "w") as f:
        f.write(content)
    print("✅ Created cookbook/measure/__init__.py")


def create_tests_init():
    """Create tests/__init__.py"""
    content = '''"""
Test Suite for ML Cookbook

Contains comprehensive tests for all components in the ML Cookbook
measurement suite.
"""

# Test configuration
TEST_CONFIG = {
    "verbose": True,
    "show_warnings": True,
    "temp_dir": "/tmp/ml_cookbook_tests"
}

__all__ = ["TEST_CONFIG"]
'''

    with open("tests/__init__.py", "w") as f:
        f.write(content)
    print("✅ Created tests/__init__.py")


def create_setup_py():
    """Create setup.py"""
    content = '''from setuptools import setup, find_packages
from pathlib import Path

def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "Professional ML Engineering Toolkit - Comprehensive measurement suite for ML experiments."

def read_requirements():
    req_path = Path(__file__).parent / "requirements.txt"
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
            return requirements
    else:
        return [
            "numpy>=1.20.0",
            "pandas>=1.3.0", 
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pyyaml>=5.4.0",
            "codecarbon>=2.1.0",
            "wandb>=0.12.0",
            "tensorboard>=2.8.0"
        ]

setup(
    name="ml-cookbook",
    version="1.0.0",
    author="Daniel Mercado",
    author_email="daniel.mercado@example.com",
    description="Professional ML Engineering Toolkit - Comprehensive measurement suite for ML experiments",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/DanielMercado60660/ml-cookbook",

    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements(),

    entry_points={
        "console_scripts": [
            "cookbook-prof=cookbook.measure.cli:main_cli",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],

    keywords=[
        "machine-learning", 
        "ml-engineering", 
        "experiment-tracking",
        "performance-profiling",
        "statistical-validation",
        "carbon-tracking",
        "mlops"
    ],

    zip_safe=False,
    platforms=["any"],
)
'''

    with open("setup.py", "w") as f:
        f.write(content)
    print("✅ Created setup.py")


def create_other_init_files():
    """Create other necessary __init__.py files"""

    # Create empty __init__.py for other directories
    other_dirs = [
        "cookbook/mechanics",
        "cookbook/models",
        "cookbook/infrastructure",
        "cookbook/applications"
    ]

    for dir_path in other_dirs:
        if Path(dir_path).exists():
            init_file = Path(dir_path) / "__init__.py"
            if not init_file.exists():
                with open(init_file, "w") as f:
                    f.write(f'"""\\n{dir_path.split("/")[1].title()} module - Coming soon!\\n"""\\n')
                print(f"✅ Created {init_file}")


def main():
    """Run the complete setup"""
    print("🔧 ML Cookbook - Setting up professional package structure...")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("cookbook").exists():
        print("❌ Error: cookbook/ directory not found!")
        print("   Make sure you're running this from the ml-cookbook root directory")
        return

    # Create all __init__.py files
    create_cookbook_init()
    create_measure_init()
    create_tests_init()
    create_other_init_files()

    # Create setup.py
    create_setup_py()

    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("✅ All __init__.py files created")
    print("✅ setup.py created")
    print("✅ Professional package structure ready")

    print("\n🚀 Next steps:")
    print("1. Test the package: python -c 'import cookbook; cookbook.measure.print_status()'")
    print("2. Install in development mode: pip install -e .")
    print("3. Test CLI: cookbook-prof --help")
    print("4. Run the full test suite: python test_migration.py")

    print("\n📦 Your package is now:")
    print("   • Installable via pip")
    print("   • Importable from anywhere")
    print("   • Professional portfolio quality")
    print("   • Ready for GitHub showcase")


if __name__ == "__main__":
    main()
''''''