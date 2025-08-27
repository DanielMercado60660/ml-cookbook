"""
Reproducibility Toolkit for ML Cookbook

This module provides utilities to ensure experiments are reliably reproducible:
- Global seed management
- Deterministic operator configurations  
- Checkpoint verification
- Repository templates with reproducibility best practices

Usage:
    from cookbook.reproduce import set_global_seed, enable_deterministic_mode
    
    # Set up reproducible environment
    set_global_seed(42)
    enable_deterministic_mode()
"""

from .seeding import set_global_seed, get_current_seed, SeedContext
from .deterministic import (
    enable_deterministic_mode,
    disable_deterministic_mode,
    get_deterministic_status,
    DeterministicConfig
)
from .verification import (
    compute_checkpoint_hash,
    verify_checkpoint_integrity,
    CheckpointVerifier
)
from .validation import validate_template_structure

__all__ = [
    # Seeding utilities
    'set_global_seed',
    'get_current_seed', 
    'SeedContext',
    
    # Deterministic operations
    'enable_deterministic_mode',
    'disable_deterministic_mode',
    'get_deterministic_status',
    'DeterministicConfig',
    
    # Verification utilities
    'compute_checkpoint_hash',
    'verify_checkpoint_integrity',
    'CheckpointVerifier',
    
    # Validation utilities
    'validate_template_structure'
]

__version__ = "0.1.0"