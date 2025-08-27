"""
Deterministic operations configuration for reproducible ML experiments.

Provides centralized control over deterministic behavior across ML frameworks,
with detailed configuration options and performance impact warnings.

Usage:
    from cookbook.reproduce import enable_deterministic_mode, DeterministicConfig
    
    # Simple deterministic mode
    enable_deterministic_mode()
    
    # Advanced configuration
    config = DeterministicConfig(
        torch_deterministic=True,
        torch_benchmark=False,
        warn_performance=True
    )
    config.apply()
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None

logger = logging.getLogger(__name__)


@dataclass
class DeterministicConfig:
    """Configuration for deterministic operations across ML frameworks."""
    
    # PyTorch settings
    torch_deterministic: bool = True
    torch_benchmark: bool = False
    torch_use_deterministic_algorithms: bool = True
    torch_warn_only: bool = False
    
    # JAX settings  
    jax_deterministic: bool = True
    jax_prng_impl: str = 'rbg'  # 'rbg' is more deterministic than 'threefry'
    
    # Performance warnings
    warn_performance: bool = True
    
    # Environment variables
    set_env_vars: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.jax_prng_impl not in ['rbg', 'threefry', 'unsafe_rbg']:
            raise ValueError(f"Invalid JAX PRNG implementation: {self.jax_prng_impl}")
    
    def apply(self) -> Dict[str, Any]:
        """
        Apply the deterministic configuration.
        
        Returns:
            Dictionary with the previous state for restoration
        """
        previous_state = {}
        
        if self.warn_performance:
            logger.warning(
                "Enabling deterministic mode may significantly impact performance. "
                "Consider disabling for production training if reproducibility is not critical."
            )
        
        # PyTorch configuration
        if TORCH_AVAILABLE:
            previous_state.update(self._configure_torch())
        else:
            logger.info("PyTorch not available, skipping PyTorch deterministic configuration")
        
        # JAX configuration  
        if JAX_AVAILABLE:
            previous_state.update(self._configure_jax())
        else:
            logger.info("JAX not available, skipping JAX deterministic configuration")
        
        # Environment variables
        if self.set_env_vars:
            previous_state.update(self._configure_env_vars())
        
        logger.info("Deterministic configuration applied successfully")
        return previous_state
    
    def _configure_torch(self) -> Dict[str, Any]:
        """Configure PyTorch for deterministic operations."""
        previous_state = {}
        
        if self.torch_deterministic:
            # Save previous state
            previous_state['torch_cudnn_deterministic'] = torch.backends.cudnn.deterministic
            previous_state['torch_cudnn_benchmark'] = torch.backends.cudnn.benchmark
            
            # Apply deterministic settings
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = self.torch_benchmark
            
            logger.info("PyTorch CUDNN deterministic mode enabled")
        
        if self.torch_use_deterministic_algorithms and hasattr(torch, 'use_deterministic_algorithms'):
            # Save previous state (if possible)
            try:
                previous_state['torch_deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
            except:
                previous_state['torch_deterministic_algorithms'] = None
            
            # Enable deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=self.torch_warn_only)
            logger.info("PyTorch deterministic algorithms enabled")
        
        return previous_state
    
    def _configure_jax(self) -> Dict[str, Any]:
        """Configure JAX for deterministic operations."""
        previous_state = {}
        
        if self.jax_deterministic:
            # Save previous PRNG implementation
            try:
                previous_state['jax_prng_impl'] = jax.config.read('jax_default_prng_impl')
            except:
                previous_state['jax_prng_impl'] = 'threefry'  # Default
            
            # Set deterministic PRNG
            jax.config.update('jax_default_prng_impl', self.jax_prng_impl)
            
            # Additional JAX deterministic settings
            jax.config.update('jax_enable_x64', True)  # More precision
            
            logger.info(f"JAX deterministic mode enabled with PRNG: {self.jax_prng_impl}")
        
        return previous_state
    
    def _configure_env_vars(self) -> Dict[str, Any]:
        """Configure environment variables for deterministic operations."""
        previous_state = {}
        
        env_vars = {
            'TF_CPP_MIN_LOG_LEVEL': '2',  # Reduce TensorFlow warnings
            'TF_DETERMINISTIC_OPS': '1',  # TensorFlow deterministic ops
            'CUDA_LAUNCH_BLOCKING': '1',  # Synchronous CUDA kernel launches
        }
        
        for key, value in env_vars.items():
            previous_state[f'env_{key}'] = os.environ.get(key)
            os.environ[key] = value
        
        logger.info("Deterministic environment variables set")
        return previous_state


def enable_deterministic_mode(config: Optional[DeterministicConfig] = None) -> Dict[str, Any]:
    """
    Enable deterministic mode across all available ML frameworks.
    
    Args:
        config: Optional custom configuration. If None, uses default settings.
        
    Returns:
        Dictionary with previous state for restoration
        
    Example:
        >>> previous_state = enable_deterministic_mode()
        >>> # Run experiments
        >>> restore_deterministic_state(previous_state)
    """
    if config is None:
        config = DeterministicConfig()
    
    return config.apply()


def disable_deterministic_mode(restore_performance: bool = True) -> None:
    """
    Disable deterministic mode to restore performance.
    
    Args:
        restore_performance: Whether to enable performance optimizations
    """
    if TORCH_AVAILABLE:
        torch.backends.cudnn.deterministic = False
        if restore_performance:
            torch.backends.cudnn.benchmark = True
        
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(False)
    
    logger.info("Deterministic mode disabled")


def restore_deterministic_state(previous_state: Dict[str, Any]) -> None:
    """
    Restore previous deterministic configuration state.
    
    Args:
        previous_state: State dictionary returned by enable_deterministic_mode()
    """
    # Restore PyTorch settings
    if TORCH_AVAILABLE:
        if 'torch_cudnn_deterministic' in previous_state:
            torch.backends.cudnn.deterministic = previous_state['torch_cudnn_deterministic']
        if 'torch_cudnn_benchmark' in previous_state:
            torch.backends.cudnn.benchmark = previous_state['torch_cudnn_benchmark']
        if 'torch_deterministic_algorithms' in previous_state and hasattr(torch, 'use_deterministic_algorithms'):
            if previous_state['torch_deterministic_algorithms'] is not None:
                torch.use_deterministic_algorithms(previous_state['torch_deterministic_algorithms'])
    
    # Restore JAX settings
    if JAX_AVAILABLE:
        if 'jax_prng_impl' in previous_state:
            jax.config.update('jax_default_prng_impl', previous_state['jax_prng_impl'])
    
    # Restore environment variables
    for key, value in previous_state.items():
        if key.startswith('env_'):
            env_key = key[4:]  # Remove 'env_' prefix
            if value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = value
    
    logger.info("Previous deterministic state restored")


def get_deterministic_status() -> Dict[str, Any]:
    """
    Get current deterministic configuration status across all frameworks.
    
    Returns:
        Dictionary with current deterministic settings
    """
    status = {}
    
    # PyTorch status
    if TORCH_AVAILABLE:
        status['torch'] = {
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
        }
        
        if hasattr(torch, 'are_deterministic_algorithms_enabled'):
            try:
                status['torch']['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
            except:
                status['torch']['deterministic_algorithms'] = 'unknown'
    
    # JAX status
    if JAX_AVAILABLE:
        try:
            status['jax'] = {
                'prng_impl': jax.config.read('jax_default_prng_impl'),
                'enable_x64': jax.config.read('jax_enable_x64'),
            }
        except:
            status['jax'] = {'status': 'unable_to_read_config'}
    
    # Environment variables
    status['environment'] = {
        'TF_DETERMINISTIC_OPS': os.environ.get('TF_DETERMINISTIC_OPS'),
        'CUDA_LAUNCH_BLOCKING': os.environ.get('CUDA_LAUNCH_BLOCKING'),
        'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
    }
    
    return status


def benchmark_deterministic_impact(func, args=(), kwargs={}, runs: int = 5) -> Dict[str, float]:
    """
    Benchmark the performance impact of deterministic mode.
    
    Args:
        func: Function to benchmark
        args: Function arguments
        kwargs: Function keyword arguments  
        runs: Number of runs for each mode
        
    Returns:
        Dictionary with timing results
    """
    import time
    
    # Benchmark with deterministic mode disabled
    disable_deterministic_mode()
    non_det_times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        non_det_times.append(time.perf_counter() - start)
    
    # Benchmark with deterministic mode enabled
    enable_deterministic_mode()
    det_times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        det_times.append(time.perf_counter() - start)
    
    avg_non_det = sum(non_det_times) / len(non_det_times)
    avg_det = sum(det_times) / len(det_times)
    
    return {
        'non_deterministic_avg': avg_non_det,
        'deterministic_avg': avg_det,
        'slowdown_factor': avg_det / avg_non_det if avg_non_det > 0 else float('inf'),
        'absolute_difference': avg_det - avg_non_det,
    }


class DeterministicContext:
    """Context manager for temporarily enabling deterministic mode."""
    
    def __init__(self, config: Optional[DeterministicConfig] = None):
        self.config = config or DeterministicConfig()
        self.previous_state = {}
    
    def __enter__(self):
        self.previous_state = self.config.apply()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        restore_deterministic_state(self.previous_state)
