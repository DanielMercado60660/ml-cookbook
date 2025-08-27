# Project 1.1: The Measurement Suite - Experiment Logger
# Clean wrapper for wandb/TensorBoard with JSONL fallback

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
import warnings
from contextlib import contextmanager

# Optional imports with graceful fallbacks
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available - using local logging only")

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("tensorboard not available - using local logging only")


@dataclass
class ExperimentConfig:
    """Configuration for experiment logging"""
    project_name: str
    experiment_name: str
    tags: List[str] = None
    notes: str = ""

    # Backend configuration
    use_wandb: bool = True
    use_tensorboard: bool = True
    use_local: bool = True  # Always keep local backup

    # Paths
    log_dir: str = "./logs"

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ExperimentLogger:
    """
    Production-quality experiment logger with multiple backends

    Provides a clean, unified interface for logging experiments with:
    - Weights & Biases integration
    - TensorBoard integration
    - Local JSONL fallback (always enabled)
    - Automatic metric aggregation and visualization
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = f"{config.experiment_name}_{int(time.time())}"
        self.metrics_buffer = []

        # Setup logging backends
        self.wandb_run = None
        self.tb_writer = None
        self.local_log_path = None

        self._setup_backends()
        self._log_experiment_start()

    def _setup_backends(self):
        """Initialize all enabled logging backends"""

        # Create log directory
        log_dir = Path(self.config.log_dir) / self.experiment_id
        log_dir.mkdir(parents=True, exist_ok=True)

        # 1. Local JSONL logging (always enabled)
        self.local_log_path = log_dir / "metrics.jsonl"

        # 2. Weights & Biases
        if self.config.use_wandb and WANDB_AVAILABLE:
            try:
                self.wandb_run = wandb.init(
                    project=self.config.project_name,
                    name=self.config.experiment_name,
                    tags=self.config.tags,
                    notes=self.config.notes,
                    reinit=True
                )
                print("✅ W&B logging enabled")
            except Exception as e:
                warnings.warn(f"W&B initialization failed: {e}")
                self.wandb_run = None

        # 3. TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            try:
                tb_log_dir = log_dir / "tensorboard"
                tb_log_dir.mkdir(exist_ok=True)
                self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
                print("✅ TensorBoard logging enabled")
            except Exception as e:
                warnings.warn(f"TensorBoard initialization failed: {e}")
                self.tb_writer = None

        print(f"📁 Local logs: {self.local_log_path}")

    def _log_experiment_start(self):
        """Log experiment configuration and metadata"""
        start_metadata = {
            "timestamp": time.time(),
            "experiment_id": self.experiment_id,
            "config": asdict(self.config),
            "event_type": "experiment_start"
        }

        # Log to local file
        self._write_local_log(start_metadata)

        # Log to W&B
        if self.wandb_run:
            wandb.config.update(asdict(self.config))

    def log_metrics(self,
                    metrics: Dict[str, float],
                    step: Optional[int] = None,
                    commit: bool = True):
        """
        Log metrics to all enabled backends

        Args:
            metrics: Dictionary of metric_name -> value
            step: Optional step number (auto-incremented if None)
            commit: Whether to commit to W&B immediately
        """
        if step is None:
            step = len(self.metrics_buffer)

        # Add metadata
        log_entry = {
            "timestamp": time.time(),
            "step": step,
            "metrics": metrics,
            "event_type": "metrics"
        }

        # Store in buffer for analysis
        self.metrics_buffer.append(log_entry)

        # Log to all backends
        self._write_local_log(log_entry)

        if self.wandb_run:
            wandb.log(metrics, step=step, commit=commit)

        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
            self.tb_writer.flush()

    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters to all backends"""
        hparam_entry = {
            "timestamp": time.time(),
            "hyperparameters": hparams,
            "event_type": "hyperparameters"
        }

        self._write_local_log(hparam_entry)

        if self.wandb_run:
            wandb.config.update(hparams)

        if self.tb_writer:
            # TensorBoard hparams require metrics, so we'll log them as text
            hparam_str = json.dumps(hparams, indent=2)
            self.tb_writer.add_text("hyperparameters", hparam_str)

    def log_artifact(self,
                     artifact_path: str,
                     artifact_type: str = "model",
                     description: str = ""):
        """Log artifacts (models, datasets, etc.)"""

        artifact_entry = {
            "timestamp": time.time(),
            "artifact_path": artifact_path,
            "artifact_type": artifact_type,
            "description": description,
            "event_type": "artifact"
        }

        self._write_local_log(artifact_entry)

        if self.wandb_run:
            artifact = wandb.Artifact(
                name=f"{artifact_type}_{self.experiment_id}",
                type=artifact_type,
                description=description
            )
            artifact.add_file(artifact_path)
            self.wandb_run.log_artifact(artifact)

    def log_gradient_norms(self, model, step: int):
        """Log gradient norms for debugging training"""
        if not hasattr(model, 'named_parameters'):
            return

        grad_norms = {}
        total_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                grad_norms[f"grad_norm/{name}"] = param_norm.item()
                total_norm += param_norm.item() ** 2

        grad_norms["grad_norm/total"] = total_norm ** 0.5

        self.log_metrics(grad_norms, step=step)

    def _write_local_log(self, entry: Dict[str, Any]):
        """Write entry to local JSONL file"""
        with open(self.local_log_path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')

    @contextmanager
    def log_section(self, section_name: str):
        """Context manager for logging timed sections"""
        start_time = time.time()
        print(f"🚀 Starting: {section_name}")

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.log_metrics({
                f"timing/{section_name}_duration_s": duration
            })
            print(f"✅ Completed: {section_name} ({duration:.2f}s)")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of logged metrics"""
        if not self.metrics_buffer:
            return {}

        import numpy as np

        # Collect all metric keys
        all_keys = set()
        for entry in self.metrics_buffer:
            all_keys.update(entry["metrics"].keys())

        summary = {}
        for key in all_keys:
            values = [
                entry["metrics"][key]
                for entry in self.metrics_buffer
                if key in entry["metrics"]
            ]

            if values:
                summary[key] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "last": values[-1]
                }

        return summary

    def finalize(self):
        """Clean up logging backends"""

        # Log experiment end
        end_metadata = {
            "timestamp": time.time(),
            "experiment_id": self.experiment_id,
            "total_metrics_logged": len(self.metrics_buffer),
            "event_type": "experiment_end"
        }
        self._write_local_log(end_metadata)

        # Close backends
        if self.wandb_run:
            wandb.finish()

        if self.tb_writer:
            self.tb_writer.close()

        print(f"🏁 Experiment logging complete: {self.experiment_id}")
        print(f"📊 Metrics logged: {len(self.metrics_buffer)}")

        # Return summary for portfolio documentation
        return self.get_metrics_summary()


# Convenience functions for quick setup
def create_logger(project_name: str,
                  experiment_name: str,
                  **kwargs) -> ExperimentLogger:
    """Quick logger creation with sensible defaults"""

    config = ExperimentConfig(
        project_name=project_name,
        experiment_name=experiment_name,
        **kwargs
    )

    return ExperimentLogger(config)


def demo_logger():
    """Demo the experiment logger"""
    print("🧪 EXPERIMENT LOGGER DEMO")
    print("=" * 50)

    # Create logger
    logger = create_logger(
        project_name="ml-cookbook",
        experiment_name="logger_demo",
        tags=["demo", "validation"],
        notes="Testing experiment logging functionality"
    )

    # Log some hyperparameters
    hparams = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model_type": "transformer",
        "optimizer": "adamw"
    }
    logger.log_hyperparameters(hparams)

    # Simulate training loop with metrics
    import numpy as np

    with logger.log_section("training_simulation"):
        for step in range(10):
            # Simulate decreasing loss, increasing accuracy
            loss = 2.0 * np.exp(-step * 0.3) + 0.1 * np.random.random()
            accuracy = 0.9 * (1 - np.exp(-step * 0.5)) + 0.05 * np.random.random()

            metrics = {
                "train/loss": loss,
                "train/accuracy": accuracy,
                "train/learning_rate": 0.001 * (0.95 ** step)  # Decay
            }

            logger.log_metrics(metrics, step=step)

            if step % 3 == 0:
                val_loss = loss + 0.1 + 0.05 * np.random.random()
                val_acc = accuracy - 0.05 + 0.02 * np.random.random()

                val_metrics = {
                    "val/loss": val_loss,
                    "val/accuracy": val_acc
                }
                logger.log_metrics(val_metrics, step=step)

    # Get summary
    summary = logger.finalize()

    print("\n📊 Metrics Summary:")
    for metric, stats in summary.items():
        print(f"   {metric}: {stats['last']:.4f} "
              f"(μ={stats['mean']:.4f}, σ={stats['std']:.4f})")

    return logger, summary


# Run demo
if __name__ == "__main__":
    demo_logger()