"""
Project structure validation utilities for reproducible ML experiments.

Provides tools to validate that ML projects follow reproducibility best practices.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)


def validate_template_structure(project_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate that a project follows reproducibility best practices.
    
    Args:
        project_path: Path to project to validate
        
    Returns:
        Dictionary with validation results
    """
    project_path = Path(project_path)
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'checks': {}
    }
    
    # Required files
    required_files = [
        'requirements.txt',
        'pyproject.toml',
        'README.md',
        '.gitignore'
    ]
    
    for file_name in required_files:
        file_path = project_path / file_name
        exists = file_path.exists()
        results['checks'][f'has_{file_name.replace(".", "_")}'] = exists
        
        if not exists:
            results['errors'].append(f"Missing required file: {file_name}")
            results['valid'] = False
    
    # Required directories
    required_dirs = ['src', 'tests']
    
    for dir_name in required_dirs:
        dir_path = project_path / dir_name
        exists = dir_path.exists() and dir_path.is_dir()
        results['checks'][f'has_{dir_name}_dir'] = exists
        
        if not exists:
            results['errors'].append(f"Missing required directory: {dir_name}")
            results['valid'] = False
    
    # Check for configuration management
    config_files = ['config.yaml', 'config.yml', 'configs/']
    has_config = any((project_path / f).exists() for f in config_files)
    results['checks']['has_configuration'] = has_config
    
    if not has_config:
        results['warnings'].append("No configuration files found. Consider adding config.yaml")
    
    # Check for reproducibility features
    src_dir = project_path / 'src'
    if src_dir.exists():
        config_py = src_dir / 'config.py'
        if config_py.exists():
            content = config_py.read_text()
            has_seed_management = 'seed' in content.lower()
            has_deterministic = 'deterministic' in content.lower()
            
            results['checks']['has_seed_management'] = has_seed_management
            results['checks']['has_deterministic_config'] = has_deterministic
            
            if not has_seed_management:
                results['warnings'].append("No seed management found in config.py")
            
            if not has_deterministic:
                results['warnings'].append("No deterministic configuration found in config.py")
    
    # Check for test coverage of reproducibility
    test_dir = project_path / 'tests'
    if test_dir.exists():
        test_repro = test_dir / 'test_reproducibility.py'
        has_repro_tests = test_repro.exists()
        results['checks']['has_reproducibility_tests'] = has_repro_tests
        
        if not has_repro_tests:
            results['warnings'].append("No reproducibility tests found")
    
    # Summary
    total_checks = len(results['checks'])
    passed_checks = sum(results['checks'].values())
    results['score'] = passed_checks / total_checks if total_checks > 0 else 0
    
    logger.info(f"Validation complete: {passed_checks}/{total_checks} checks passed")
    
    return results