"""
Checkpoint and artifact verification utilities for reproducible ML experiments.

Provides tools to verify the integrity of model checkpoints, datasets, and other
experimental artifacts using cryptographic hashes and metadata validation.

Usage:
    from cookbook.reproduce import compute_checkpoint_hash, CheckpointVerifier
    
    # Simple hash computation
    hash_value = compute_checkpoint_hash("model.pth")
    
    # Advanced verification with metadata
    verifier = CheckpointVerifier("experiments/")
    verifier.register_checkpoint("model.pth", metadata={"epoch": 10})
    is_valid = verifier.verify_checkpoint("model.pth")
"""

import hashlib
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import pickle
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass 
class CheckpointMetadata:
    """Metadata for checkpoint verification."""
    file_path: str
    file_size: int
    hash_algorithm: str
    hash_value: str
    created_timestamp: float
    last_modified: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create instance from dictionary."""
        return cls(**data)


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256", chunk_size: int = 8192) -> str:
    """
    Compute cryptographic hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        chunk_size: Size of chunks to read at a time (for large files)
        
    Returns:
        Hexadecimal hash string
        
    Example:
        >>> hash_val = compute_file_hash("model.pth", "sha256")
        >>> print(f"SHA256: {hash_val}")
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Initialize hash object
    hash_obj = hashlib.new(algorithm)
    
    # Read file in chunks to handle large files efficiently
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def compute_checkpoint_hash(checkpoint_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute hash of a model checkpoint with framework-specific optimizations.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        algorithm: Hash algorithm to use
        
    Returns:
        Hexadecimal hash string
        
    Example:
        >>> hash_val = compute_checkpoint_hash("model.pth")
    """
    checkpoint_path = Path(checkpoint_path)
    
    # For PyTorch checkpoints, we can be more intelligent about what we hash
    if TORCH_AVAILABLE and checkpoint_path.suffix in ['.pth', '.pt']:
        return _compute_torch_checkpoint_hash(checkpoint_path, algorithm)
    else:
        return compute_file_hash(checkpoint_path, algorithm)


def _compute_torch_checkpoint_hash(checkpoint_path: Path, algorithm: str) -> str:
    """Compute hash of PyTorch checkpoint with state dict focus."""
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create a deterministic representation
        if isinstance(checkpoint, dict):
            # Sort keys for deterministic ordering
            hash_obj = hashlib.new(algorithm)
            
            for key in sorted(checkpoint.keys()):
                # Hash the key
                hash_obj.update(str(key).encode())
                
                # Hash the value based on type
                value = checkpoint[key]
                if hasattr(value, 'numpy'):  # PyTorch tensor
                    hash_obj.update(value.detach().cpu().numpy().tobytes())
                elif isinstance(value, (int, float, str, bool)):
                    hash_obj.update(str(value).encode())
                elif isinstance(value, (list, tuple)):
                    hash_obj.update(str(value).encode())
                else:
                    # Fallback to pickle for other types
                    hash_obj.update(pickle.dumps(value))
            
            return hash_obj.hexdigest()
        else:
            # Fallback to file hash for non-dict checkpoints
            return compute_file_hash(checkpoint_path, algorithm)
            
    except Exception as e:
        logger.warning(f"Failed to compute structured hash for {checkpoint_path}: {e}. Using file hash.")
        return compute_file_hash(checkpoint_path, algorithm)


def verify_checkpoint_integrity(checkpoint_path: Union[str, Path], expected_hash: str, algorithm: str = "sha256") -> bool:
    """
    Verify checkpoint integrity against expected hash.
    
    Args:
        checkpoint_path: Path to checkpoint file
        expected_hash: Expected hash value
        algorithm: Hash algorithm used
        
    Returns:
        True if integrity check passes, False otherwise
        
    Example:
        >>> is_valid = verify_checkpoint_integrity("model.pth", "abc123...")
    """
    try:
        actual_hash = compute_checkpoint_hash(checkpoint_path, algorithm)
        is_valid = actual_hash == expected_hash
        
        if is_valid:
            logger.info(f"Checkpoint integrity verified: {checkpoint_path}")
        else:
            logger.error(f"Checkpoint integrity failed: {checkpoint_path}")
            logger.error(f"Expected: {expected_hash}")
            logger.error(f"Actual: {actual_hash}")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error verifying checkpoint integrity: {e}")
        return False


class CheckpointVerifier:
    """
    Manages checkpoint verification for a directory of experimental artifacts.
    
    Maintains a verification database with checksums and metadata for all
    registered checkpoints and artifacts.
    """
    
    def __init__(self, base_directory: Union[str, Path], verification_file: str = ".checkpoint_verification.json"):
        """
        Initialize checkpoint verifier.
        
        Args:
            base_directory: Base directory for experiments
            verification_file: Name of verification database file
        """
        self.base_directory = Path(base_directory)
        self.verification_file = self.base_directory / verification_file
        self.metadata_db: Dict[str, CheckpointMetadata] = {}
        
        # Create base directory if it doesn't exist
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
        # Load existing verification database
        self.load_verification_db()
    
    def register_checkpoint(self, 
                          checkpoint_path: Union[str, Path], 
                          metadata: Optional[Dict[str, Any]] = None,
                          algorithm: str = "sha256") -> str:
        """
        Register a checkpoint for verification.
        
        Args:
            checkpoint_path: Path to checkpoint (relative to base_directory)
            metadata: Additional metadata to store
            algorithm: Hash algorithm to use
            
        Returns:
            Computed hash value
        """
        checkpoint_path = Path(checkpoint_path)
        full_path = self.base_directory / checkpoint_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {full_path}")
        
        # Compute hash
        hash_value = compute_checkpoint_hash(full_path, algorithm)
        
        # Get file stats
        stat = full_path.stat()
        
        # Create metadata
        checkpoint_metadata = CheckpointMetadata(
            file_path=str(checkpoint_path),
            file_size=stat.st_size,
            hash_algorithm=algorithm,
            hash_value=hash_value,
            created_timestamp=time.time(),
            last_modified=stat.st_mtime,
            metadata=metadata or {}
        )
        
        # Store in database
        self.metadata_db[str(checkpoint_path)] = checkpoint_metadata
        
        # Save database
        self.save_verification_db()
        
        logger.info(f"Registered checkpoint: {checkpoint_path} (hash: {hash_value[:12]}...)")
        return hash_value
    
    def verify_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Verify a registered checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (relative to base_directory)
            
        Returns:
            True if verification passes, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)
        str_path = str(checkpoint_path)
        
        if str_path not in self.metadata_db:
            logger.error(f"Checkpoint not registered: {checkpoint_path}")
            return False
        
        metadata = self.metadata_db[str_path]
        full_path = self.base_directory / checkpoint_path
        
        if not full_path.exists():
            logger.error(f"Checkpoint file missing: {full_path}")
            return False
        
        # Verify file hasn't changed
        stat = full_path.stat()
        if stat.st_mtime != metadata.last_modified:
            logger.warning(f"Checkpoint modification time changed: {checkpoint_path}")
        
        if stat.st_size != metadata.file_size:
            logger.error(f"Checkpoint file size changed: {checkpoint_path}")
            return False
        
        # Verify hash
        return verify_checkpoint_integrity(
            full_path, 
            metadata.hash_value, 
            metadata.hash_algorithm
        )
    
    def verify_all_checkpoints(self) -> Dict[str, bool]:
        """
        Verify all registered checkpoints.
        
        Returns:
            Dictionary mapping checkpoint paths to verification results
        """
        results = {}
        
        for checkpoint_path in self.metadata_db.keys():
            results[checkpoint_path] = self.verify_checkpoint(checkpoint_path)
        
        # Summary logging
        total = len(results)
        passed = sum(results.values())
        failed = total - passed
        
        logger.info(f"Verification complete: {passed}/{total} passed, {failed} failed")
        
        return results
    
    def get_checkpoint_metadata(self, checkpoint_path: Union[str, Path]) -> Optional[CheckpointMetadata]:
        """Get metadata for a registered checkpoint."""
        return self.metadata_db.get(str(checkpoint_path))
    
    def list_registered_checkpoints(self) -> List[str]:
        """List all registered checkpoint paths."""
        return list(self.metadata_db.keys())
    
    def remove_checkpoint(self, checkpoint_path: Union[str, Path]) -> bool:
        """
        Remove a checkpoint from verification database.
        
        Args:
            checkpoint_path: Path to checkpoint
            
        Returns:
            True if removed, False if not found
        """
        str_path = str(checkpoint_path)
        if str_path in self.metadata_db:
            del self.metadata_db[str_path]
            self.save_verification_db()
            logger.info(f"Removed checkpoint from verification: {checkpoint_path}")
            return True
        return False
    
    def save_verification_db(self) -> None:
        """Save verification database to disk."""
        db_data = {
            path: metadata.to_dict() 
            for path, metadata in self.metadata_db.items()
        }
        
        with open(self.verification_file, 'w') as f:
            json.dump(db_data, f, indent=2, sort_keys=True)
    
    def load_verification_db(self) -> None:
        """Load verification database from disk."""
        if self.verification_file.exists():
            try:
                with open(self.verification_file, 'r') as f:
                    db_data = json.load(f)
                
                self.metadata_db = {
                    path: CheckpointMetadata.from_dict(metadata_dict)
                    for path, metadata_dict in db_data.items()
                }
                
                logger.info(f"Loaded verification database with {len(self.metadata_db)} entries")
                
            except Exception as e:
                logger.error(f"Failed to load verification database: {e}")
                self.metadata_db = {}
        else:
            self.metadata_db = {}
    
    def export_verification_report(self, output_path: Union[str, Path]) -> None:
        """
        Export a human-readable verification report.
        
        Args:
            output_path: Path for the report file
        """
        verification_results = self.verify_all_checkpoints()
        
        report_lines = [
            "# Checkpoint Verification Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Base Directory: {self.base_directory}",
            "",
            "## Summary",
            f"Total Checkpoints: {len(verification_results)}",
            f"Verified: {sum(verification_results.values())}",
            f"Failed: {sum(1 for v in verification_results.values() if not v)}",
            "",
            "## Checkpoint Details"
        ]
        
        for checkpoint_path, is_valid in verification_results.items():
            metadata = self.metadata_db[checkpoint_path]
            status = "✅ VALID" if is_valid else "❌ FAILED"
            
            report_lines.extend([
                f"### {checkpoint_path}",
                f"Status: {status}",
                f"Hash: {metadata.hash_value}",
                f"Algorithm: {metadata.hash_algorithm}",
                f"File Size: {metadata.file_size:,} bytes",
                f"Last Modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata.last_modified))}",
                ""
            ])
            
            if metadata.metadata:
                report_lines.append("Metadata:")
                for key, value in metadata.metadata.items():
                    report_lines.append(f"  - {key}: {value}")
                report_lines.append("")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Verification report exported to: {output_path}")


def create_integrity_manifest(directory: Union[str, Path], 
                            output_file: str = "integrity_manifest.json",
                            patterns: List[str] = None) -> Dict[str, str]:
    """
    Create an integrity manifest for all files matching patterns in a directory.
    
    Args:
        directory: Directory to scan
        output_file: Output manifest file name
        patterns: File patterns to include (e.g., ['*.pth', '*.json'])
        
    Returns:
        Dictionary mapping file paths to hash values
    """
    directory = Path(directory)
    patterns = patterns or ['*.pth', '*.pt', '*.pkl', '*.json', '*.yaml', '*.yml']
    
    manifest = {}
    
    for pattern in patterns:
        for file_path in directory.rglob(pattern):
            if file_path.is_file():
                relative_path = file_path.relative_to(directory)
                hash_value = compute_file_hash(file_path)
                manifest[str(relative_path)] = hash_value
    
    # Save manifest
    manifest_path = directory / output_file
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    
    logger.info(f"Created integrity manifest with {len(manifest)} files: {manifest_path}")
    return manifest
