"""Configuration management for fine-tuning framework.

This module handles loading, saving, and managing training configurations
including model settings, training hyperparameters, data paths, and logging.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Model configuration settings."""

    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    model_cache_dir: Optional[str] = None
    use_flash_attention: bool = True
    torch_dtype: str = "auto"  # "auto", "float16", "bfloat16", "float32"
    device_map: str = "auto"
    trust_remote_code: bool = True

    # PEFT/LoRA settings
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Quantization (QLoRA)
    load_in_8bit: bool = False
    load_in_4bit: bool = False


@dataclass
class DataConfig:
    """Data configuration settings."""

    data_path: str = "datasets/spatial_data.jsonl"
    task: str = "A"  # Task A: Description -> ASCII
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_samples: Optional[int] = None  # Limit dataset size for testing
    seed: int = 42

    # Task A specific settings
    ascii_view: str = "grid"  # "grid", "simple", or "panel"
    max_description_length: int = 512
    max_seq_length: Optional[int] = (
        None  # Caps SFT sequence length; required for packing
    )

    # Data filtering
    require_ascii: bool = True
    min_num_components: Optional[int] = None
    max_num_components: Optional[int] = None

    # Format settings
    prompt_completion_format: bool = (
        False  # Use prompt/completion format instead of messages
    )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    output_dir: str = "outputs/task_a_finetuning"
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Optimizer settings
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"

    # Evaluation settings
    eval_steps: int = 100
    save_strategy: str = "epoch"  # "epoch" or "steps"
    save_steps: Optional[int] = (
        None  # Number of training steps between checkpoints (only used when save_strategy="steps")
    )
    logging_steps: int = 10
    save_total_limit: int = 3

    # Mixed precision training
    fp16: bool = False
    bf16: bool = True

    # Other settings
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    completion_only_loss: bool = False  # Compute loss only on completion part

    # Memory/throughput knobs
    gradient_checkpointing: bool = False
    packing: bool = False


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration."""

    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True

    # Experiment tracking
    use_wandb: bool = False
    wandb_project: Optional[str] = "spatial-reasoning-finetuning"
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: Optional[str] = None


@dataclass
class Config:
    """Main configuration class combining all settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Experiment metadata
    description: str = "Fine-tuning Qwen3-30B on Task A (Description -> ASCII)"

    def __post_init__(self):
        """Validate and setup paths after initialization."""
        # Create output directories
        os.makedirs(self.training.output_dir, exist_ok=True)
        os.makedirs(self.logging.log_dir, exist_ok=True)

        # Set tensorboard dir if not specified
        if self.logging.use_tensorboard and self.logging.tensorboard_dir is None:
            self.logging.tensorboard_dir = os.path.join(
                self.training.output_dir, "tensorboard"
            )
            os.makedirs(self.logging.tensorboard_dir, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": asdict(self.model),
            "data": asdict(self.data),
            "training": asdict(self.training),
            "logging": asdict(self.logging),
            "description": self.description,
        }

    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)

        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            description=config_dict.get("description", ""),
        )

    @classmethod
    def from_args(cls, args) -> "Config":
        """Create config from command-line arguments."""
        config = cls()

        # Override with command-line arguments if provided
        if hasattr(args, "model_name") and args.model_name:
            config.model.model_name = args.model_name
        if hasattr(args, "data_path") and args.data_path:
            config.data.data_path = args.data_path
        if hasattr(args, "output_dir") and args.output_dir:
            config.training.output_dir = args.output_dir
        if hasattr(args, "num_epochs") and args.num_epochs:
            config.training.num_epochs = args.num_epochs
        if hasattr(args, "batch_size") and args.batch_size:
            config.training.batch_size = args.batch_size
        if hasattr(args, "learning_rate") and args.learning_rate:
            config.training.learning_rate = args.learning_rate

        return config


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
