"""Logging and monitoring utilities for Task A fine-tuning.

This module handles logging setup, experiment tracking with WandB and TensorBoard,
and monitoring training progress.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """Unified logger for training with support for console, file, WandB, and TensorBoard."""

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        log_to_file: bool = True,
        log_to_console: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        tensorboard_dir: Optional[str] = None,
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            use_wandb: Whether to use Weights & Biases
            wandb_project: WandB project name
            wandb_run_name: WandB run name
            wandb_entity: WandB entity/team name
            wandb_config: Configuration dict to log to WandB
            use_tensorboard: Whether to use TensorBoard
            tensorboard_dir: TensorBoard log directory
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup Python logger
        self.logger = self._setup_logger(
            log_level=log_level,
            log_to_file=log_to_file,
            log_to_console=log_to_console,
        )

        # Setup WandB
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_run = None
        if self.use_wandb:
            self._setup_wandb(
                project=wandb_project,
                run_name=wandb_run_name,
                entity=wandb_entity,
                config=wandb_config,
            )
        elif use_wandb and not WANDB_AVAILABLE:
            self.logger.warning(
                "WandB requested but not available. Install with: pip install wandb"
            )

        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.tb_writer = None
        if self.use_tensorboard:
            self._setup_tensorboard(tensorboard_dir)
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            self.logger.warning(
                "TensorBoard requested but not available. Install with: pip install tensorboard"
            )

    def _setup_logger(
        self,
        log_level: str,
        log_to_file: bool,
        log_to_console: bool,
    ) -> logging.Logger:
        """Setup Python logger."""
        logger = logging.getLogger("training")
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"training_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")

        return logger

    def _setup_wandb(
        self,
        project: Optional[str],
        run_name: Optional[str],
        entity: Optional[str],
        config: Optional[Dict[str, Any]],
    ):
        """Setup Weights & Biases logging."""
        try:
            self.wandb_run = wandb.init(
                project=project or "spatial-reasoning-finetuning",
                name=run_name,
                entity=entity,
                config=config,
                reinit=True,
            )
            self.logger.info(f"WandB initialized: {self.wandb_run.url}")
        except Exception as e:
            self.logger.error(f"Failed to initialize WandB: {e}")
            self.use_wandb = False

    def _setup_tensorboard(self, tensorboard_dir: Optional[str]):
        """Setup TensorBoard logging."""
        try:
            if tensorboard_dir is None:
                tensorboard_dir = self.log_dir / "tensorboard"

            tb_path = Path(tensorboard_dir)
            tb_path.mkdir(parents=True, exist_ok=True)

            self.tb_writer = SummaryWriter(log_dir=str(tb_path))
            self.logger.info(f"TensorBoard logging to: {tb_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard: {e}")
            self.use_tensorboard = False

    def log(self, message: str, level: str = "INFO"):
        """Log a message.

        Args:
            message: Message to log
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        log_func = getattr(self.logger, level.lower())
        log_func(message)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric names and values
            step: Training step/iteration
            prefix: Prefix to add to metric names
        """
        # Log to console/file
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )
        step_str = f"Step {step} - " if step is not None else ""
        self.logger.info(f"{step_str}{prefix}{metrics_str}")

        # Log to WandB
        if self.use_wandb:
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
            if step is not None:
                wandb.log(wandb_metrics, step=step)
            else:
                wandb.log(wandb_metrics)

        # Log to TensorBoard
        if self.use_tensorboard and step is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}{name}", value, step)

    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters.

        Args:
            params: Dictionary of hyperparameters
        """
        self.logger.info("Hyperparameters:")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

        if self.use_wandb:
            wandb.config.update(params)

    def log_model_info(self, model):
        """Log model information.

        Args:
            model: Model to log information about
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

        if self.use_wandb:
            wandb.config.update(
                {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "trainable_percentage": 100 * trainable_params / total_params,
                }
            )

    def finish(self):
        """Finish logging and cleanup."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()

        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()

        self.logger.info("Logging finished")


def setup_logger_from_config(config) -> TrainingLogger:
    """Setup logger from configuration object.

    Args:
        config: Configuration object with logging settings

    Returns:
        TrainingLogger instance
    """
    return TrainingLogger(
        log_dir=config.logging.log_dir,
        log_level=config.logging.log_level,
        log_to_file=config.logging.log_to_file,
        log_to_console=config.logging.log_to_console,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        wandb_run_name=config.logging.wandb_run_name,
        wandb_entity=config.logging.wandb_entity,
        wandb_config=config.to_dict(),
        use_tensorboard=config.logging.use_tensorboard,
        tensorboard_dir=config.logging.tensorboard_dir,
    )
