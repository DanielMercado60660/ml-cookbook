"""
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
        CarbonTracker,
        CookbookCLI, CarbonTracker
)

    __all__ = [
        "PerformanceProfiler",
        "ExperimentLogger",
        "ExperimentConfig", 
        "StatisticalValidator",
        "TestType",
        "EffectSize",
        "CarbonTracker",
        "CookbookCLI",
        "__version__",
        "__author__",
        "__description__"
    ]
except ImportError as e:
    print(f"Warning: Some components not available - {e}")
    __all__ = ["__version__", "__author__", "__description__"]
