#!/usr/bin/env python3
"""
Command-line interface for ML Cookbook Reproducibility Toolkit.

Usage:
    python -m cookbook.reproduce --help
    python -m cookbook.reproduce seed --set 42
    python -m cookbook.reproduce verify --checkpoint model.pth
    python -m cookbook.reproduce template --create my_project
"""

import argparse
import sys
from pathlib import Path

def cmd_seed(args):
    """Handle seed-related commands."""
    from cookbook.reproduce import set_global_seed, get_current_seed
    
    if args.set:
        set_global_seed(args.set)
        print(f"Global seed set to: {args.set}")
    
    current = get_current_seed()
    if current is not None:
        print(f"Current seed: {current}")
    else:
        print("No global seed set")

def cmd_verify(args):
    """Handle verification commands."""
    from cookbook.reproduce import compute_checkpoint_hash, CheckpointVerifier
    
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return 1
        
        hash_value = compute_checkpoint_hash(checkpoint_path)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Hash: {hash_value}")
        
    if args.directory:
        verifier = CheckpointVerifier(args.directory)
        results = verifier.verify_all_checkpoints()
        
        print(f"Verification results for {args.directory}:")
        for checkpoint, is_valid in results.items():
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"  {checkpoint}: {status}")
        
        total = len(results)
        valid = sum(results.values())
        print(f"\nSummary: {valid}/{total} checkpoints valid")
        
        return 0 if valid == total else 1
    
    return 0

def cmd_template(args):
    """Handle template commands."""
    from cookbook.reproduce import create_reproducible_template, TemplateConfig, validate_template_structure
    
    if args.create:
        project_path = Path(args.create)
        
        config = TemplateConfig(
            project_name=project_path.name,
            author=args.author or "ML Cookbook User",
            python_version=args.python_version or "3.10",
            frameworks=args.frameworks.split(',') if args.frameworks else ['pytorch'],
            include_docker=args.docker,
            include_uv_lock=args.uv
        )
        
        success = create_reproducible_template(project_path, config)
        
        if success:
            print(f"✅ Created reproducible project: {project_path}")
            print(f"Next steps:")
            print(f"  cd {project_path}")
            print(f"  pip install -r requirements.txt")
            print(f"  python -m {project_path.name}")
        else:
            print(f"❌ Failed to create project: {project_path}")
            return 1
    
    if args.validate:
        project_path = Path(args.validate)
        
        if not project_path.exists():
            print(f"Project not found: {project_path}")
            return 1
        
        validation = validate_template_structure(project_path)
        
        print(f"Validation results for {project_path}:")
        print(f"Score: {validation['score']:.2f}")
        print(f"Status: {'✅ VALID' if validation['valid'] else '❌ INVALID'}")
        
        if validation['errors']:
            print("\nErrors:")
            for error in validation['errors']:
                print(f"  - {error}")
        
        if validation['warnings']:
            print("\nWarnings:")
            for warning in validation['warnings']:
                print(f"  - {warning}")
        
        return 0 if validation['valid'] else 1
    
    return 0

def cmd_deterministic(args):
    """Handle deterministic mode commands."""
    from cookbook.reproduce import enable_deterministic_mode, disable_deterministic_mode, get_deterministic_status
    
    if args.enable:
        enable_deterministic_mode()
        print("✅ Deterministic mode enabled")
    
    if args.disable:
        disable_deterministic_mode()
        print("✅ Deterministic mode disabled")
    
    if args.status:
        status = get_deterministic_status()
        print("Deterministic status:")
        for framework, settings in status.items():
            if isinstance(settings, dict):
                print(f"  {framework}:")
                for key, value in settings.items():
                    if value is not None:
                        print(f"    {key}: {value}")
            else:
                print(f"  {framework}: {settings}")
    
    return 0

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ML Cookbook Reproducibility Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set global seed
  python -m cookbook.reproduce seed --set 42
  
  # Verify checkpoint
  python -m cookbook.reproduce verify --checkpoint model.pth
  
  # Create reproducible project
  python -m cookbook.reproduce template --create my_experiment
  
  # Validate project structure  
  python -m cookbook.reproduce template --validate my_experiment
  
  # Enable deterministic mode
  python -m cookbook.reproduce deterministic --enable --status
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Seed commands
    seed_parser = subparsers.add_parser('seed', help='Manage global seeds')
    seed_parser.add_argument('--set', type=int, help='Set global seed')
    
    # Verification commands
    verify_parser = subparsers.add_parser('verify', help='Verify checkpoints')
    verify_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    verify_parser.add_argument('--directory', type=str, help='Verify all checkpoints in directory')
    
    # Template commands
    template_parser = subparsers.add_parser('template', help='Manage project templates')
    template_parser.add_argument('--create', type=str, help='Create new reproducible project')
    template_parser.add_argument('--validate', type=str, help='Validate project structure')
    template_parser.add_argument('--author', type=str, help='Project author name')
    template_parser.add_argument('--python-version', type=str, help='Python version (default: 3.10)')
    template_parser.add_argument('--frameworks', type=str, help='Frameworks to include (comma-separated)')
    template_parser.add_argument('--docker', action='store_true', help='Include Docker support')
    template_parser.add_argument('--uv', action='store_true', help='Include uv.lock file')
    
    # Deterministic commands
    det_parser = subparsers.add_parser('deterministic', help='Manage deterministic mode')
    det_parser.add_argument('--enable', action='store_true', help='Enable deterministic mode')
    det_parser.add_argument('--disable', action='store_true', help='Disable deterministic mode')
    det_parser.add_argument('--status', action='store_true', help='Show deterministic status')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'seed':
            return cmd_seed(args)
        elif args.command == 'verify':
            return cmd_verify(args)
        elif args.command == 'template':
            return cmd_template(args)
        elif args.command == 'deterministic':
            return cmd_deterministic(args)
        else:
            parser.print_help()
            return 1
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the cookbook.reproduce module is installed and available")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
