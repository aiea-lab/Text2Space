"""Utility functions for Task A fine-tuning.

This module provides helper functions for checkpoint management,
metrics computation, and other common operations.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def cleanup_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
    checkpoint_prefix: str = "checkpoint",
):
    """Remove old checkpoints, keeping only the last N.

    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        checkpoint_prefix: Prefix of checkpoint files/directories
    """
    if not os.path.exists(output_dir):
        return

    # Find all checkpoint files
    checkpoint_files = []
    for item in os.listdir(output_dir):
        if item.startswith(checkpoint_prefix):
            path = os.path.join(output_dir, item)
            checkpoint_files.append((path, os.path.getmtime(path)))

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)

    # Remove old checkpoints
    for path, _ in checkpoint_files[keep_last_n:]:
        if os.path.isfile(path):
            os.remove(path)
            print(f"Removed old checkpoint: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path)
            print(f"Removed old checkpoint directory: {path}")


def save_predictions(
    predictions: List[Dict[str, Any]],
    output_path: str,
    format: str = "jsonl",
):
    """Save predictions to file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save predictions
        format: Output format ("jsonl" or "json")
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if format == "jsonl":
        with open(output_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
    elif format == "json":
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Predictions saved to {output_path}")


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
