"""
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
    from .carbon import CarbonTracker, CarbonMetrics, CarbonAwareProfiler
    __all__.extend(["CarbonTracker", "CarbonMetrics", "CarbonAwareProfiler"])
    CARBON_AVAILABLE = True
except ImportError as e:
    CARBON_AVAILABLE = False
    print(f"Warning: CarbonTracker not available - {e}")

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
