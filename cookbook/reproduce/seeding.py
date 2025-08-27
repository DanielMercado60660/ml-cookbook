"""
Global seeding utilities for reproducible ML experiments.

Provides centralized seed management across all major ML libraries:
- Python's random module
- NumPy 
- PyTorch (CPU and CUDA)
- JAX
- OS environment variables

Usage:
    from cookbook.reproduce import set_global_seed, SeedContext
    
    # Set global seed for entire experiment
    set_global_seed(42)
    
    # Temporarily change seed for specific operation
    with SeedContext(123):
        # Operations with seed 123
        pass
    # Back to original seed
"""

import random
import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global seed state
_current_seed: Optional[int] = None
_seed_history: list = []


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set global seed across all ML libraries for reproducible experiments.
    
    Args:
        seed: Integer seed value
        deterministic: Whether to enable deterministic operations (may impact performance)
        
    Example:
        >>> set_global_seed(42)
        >>> # All random operations now use seed 42
    """
    global _current_seed
    
    logger.info(f"Setting global seed to {seed}")
    
    # Python's random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # OS environment (for some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            # Enable deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # For newer PyTorch versions
            if hasattr(torch, 'use_deterministic_algorithms'):
                torch.use_deterministic_algorithms(True)
    
    # JAX
    if JAX_AVAILABLE:
        # JAX handles seeds differently - we'll set a base seed
        jax.config.update('jax_default_prng_impl', 'rbg')
        os.environ['JAX_SEED'] = str(seed)
    
    # Update global state
    _current_seed = seed
    _seed_history.append({
        'seed': seed,
        'timestamp': np.datetime64('now'),
        'deterministic': deterministic
    })
    
    logger.info(f"Global seed set to {seed} (deterministic={deterministic})")


def get_current_seed() -> Optional[int]:
    """Get the currently active global seed."""
    return _current_seed


def get_seed_history() -> list:
    """Get the history of all seeds set during this session."""
    return _seed_history.copy()


@contextmanager
def SeedContext(seed: int, deterministic: bool = True):
    """
    Context manager for temporarily changing the global seed.
    
    Args:
        seed: Temporary seed to use
        deterministic: Whether to enable deterministic operations
        
    Example:
        >>> set_global_seed(42)
        >>> with SeedContext(123):
        ...     # Operations use seed 123
        ...     data = np.random.rand(10)
        >>> # Back to seed 42
    """
    # Save current state
    original_seed = _current_seed
    original_random_state = random.getstate()
    original_numpy_state = np.random.get_state()
    
    original_torch_state = None
    original_torch_cuda_state = None
    if TORCH_AVAILABLE:
        original_torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            original_torch_cuda_state = torch.cuda.get_rng_state_all()
    
    try:
        # Set temporary seed
        set_global_seed(seed, deterministic=deterministic)
        yield
    finally:
        # Restore original state
        if original_seed is not None:
            random.setstate(original_random_state)
            np.random.set_state(original_numpy_state)
            
            if TORCH_AVAILABLE:
                torch.set_rng_state(original_torch_state)
                if torch.cuda.is_available() and original_torch_cuda_state is not None:
                    torch.cuda.set_rng_state_all(original_torch_cuda_state)
            
            _current_seed = original_seed
            logger.info(f"Restored original seed {original_seed}")


def save_seed_state(filepath: str) -> None:
    """
    Save current random number generator states to file.
    
    Args:
        filepath: Path to save the state file
    """
    state = {
        'current_seed': _current_seed,
        'random_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'seed_history': _seed_history
    }
    
    if TORCH_AVAILABLE:
        state['torch_state'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda_state'] = torch.cuda.get_rng_state_all()
    
    np.save(filepath, state, allow_pickle=True)
    logger.info(f"Saved seed state to {filepath}")


def load_seed_state(filepath: str) -> None:
    """
    Load random number generator states from file.
    
    Args:
        filepath: Path to load the state file from
    """
    global _current_seed, _seed_history
    
    state = np.load(filepath, allow_pickle=True).item()
    
    _current_seed = state['current_seed']
    _seed_history = state['seed_history']
    
    random.setstate(state['random_state'])
    np.random.set_state(state['numpy_state'])
    
    if TORCH_AVAILABLE and 'torch_state' in state:
        torch.set_rng_state(state['torch_state'])
        if torch.cuda.is_available() and 'torch_cuda_state' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda_state'])
    
    logger.info(f"Loaded seed state from {filepath}")


def generate_deterministic_seeds(base_seed: int, count: int) -> list:
    """
    Generate a deterministic sequence of seeds from a base seed.
    
    Useful for distributed training where each worker needs a unique but deterministic seed.
    
    Args:
        base_seed: Base seed to derive from
        count: Number of seeds to generate
        
    Returns:
        List of deterministic seeds
        
    Example:
        >>> seeds = generate_deterministic_seeds(42, 4)
        >>> # Use seeds[0] for worker 0, seeds[1] for worker 1, etc.
    """
    rng = np.random.RandomState(base_seed)
    return [int(rng.randint(0, 2**31 - 1)) for _ in range(count)]


def verify_deterministic_execution(func, *args, **kwargs) -> bool:
    """
    Verify that a function produces deterministic outputs across multiple runs.
    
    Args:
        func: Function to test
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        True if function is deterministic, False otherwise
        
    Example:
        >>> def my_model_forward(x):
        ...     return model(x)
        >>> is_deterministic = verify_deterministic_execution(my_model_forward, input_tensor)
    """
    if _current_seed is None:
        raise ValueError("No global seed set. Call set_global_seed() first.")
    
    # Run function 3 times with same seed
    results = []
    for i in range(3):
        with SeedContext(_current_seed):
            result = func(*args, **kwargs)
            if hasattr(result, 'detach'):  # PyTorch tensor
                result = result.detach().cpu().numpy()
            elif hasattr(result, 'numpy'):  # JAX array
                result = np.array(result)
            results.append(result)
    
    # Check if all results are identical
    for i in range(1, len(results)):
        if not np.allclose(results[0], results[i], atol=1e-10):
            logger.warning(f"Non-deterministic behavior detected. Run {i} differs from run 0.")
            return False
    
    logger.info("Function execution is deterministic")
    return True


# Export this function explicitly
__all__ = [
    'set_global_seed',
    'get_current_seed',
    'get_seed_history',
    'SeedContext',
    'save_seed_state',
    'load_seed_state',
    'generate_deterministic_seeds',
    'verify_deterministic_execution'
]
