"""Dataset package for transfer learning tasks.

This package provides dataset classes for different tasks:
- TaskADataset: Description -> ASCII generation
- TaskARandomFormatDataset: Description -> ASCII generation with random formats
- TaskBDataset: ASCII -> Description generation
- MixedDataset: 50/50 mix of Task A and Task B
"""

from typing import Optional, Tuple

from datasets import Dataset as HFDataset

from .base_dataset import BaseDataset
from .mixed_dataset import MixedDataset
from .task_a_dataset import TaskADataset
from .task_a_random_format_dataset import TaskARandomFormatDataset
from .task_b_dataset import TaskBDataset

__all__ = [
    "BaseDataset",
    "TaskADataset",
    "TaskARandomFormatDataset",
    "TaskBDataset",
    "MixedDataset",
    "prepare_datasets",
]


def prepare_datasets(
    data_path: str,
    task: str = "A",
    ascii_view: str = "grid",
    max_description_length: int = 512,
    require_ascii: bool = True,
    min_num_components: Optional[int] = None,
    max_num_components: Optional[int] = None,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42,
    prompt_completion_format: bool = False,
) -> Tuple[HFDataset, HFDataset, HFDataset]:
    """Prepare train, validation, and test datasets based on task type.

    Args:
        data_path: Path to JSONL data file
        task: Task type - "A" (Description->ASCII), "B" (ASCII->Description), or "mixed" (A+B)
        ascii_view: Which ASCII view to use ("grid", "simple", or "panel")
        max_description_length: Maximum length for description text
        require_ascii: Filter out samples without ASCII
        min_num_components: Minimum number of components to include
        max_num_components: Maximum number of components to include
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        max_samples: Maximum number of samples to load (for testing)
        seed: Random seed for reproducibility
        prompt_completion_format: Use prompt/completion format instead of messages

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as HuggingFace Datasets

    Raises:
        ValueError: If task is not one of "A", "B", or "mixed", or "A-RANDOM-FORMAT".
    """
    # Normalize task name
    task = task.upper()

    # Select appropriate dataset class
    if task == "A":
        dataset_class = TaskADataset
    elif task == "B":
        dataset_class = TaskBDataset
    elif task == "MIXED" or task == "A+B":
        dataset_class = MixedDataset
    elif task == "A-RANDOM-FORMAT":
        dataset_class = TaskARandomFormatDataset
    else:
        raise ValueError(
            f"Invalid task '{task}'. Must be one of: 'A', 'B', 'mixed', or 'A+B', or 'A-RANDOM-FORMAT'."
        )

    # Create dataset instance
    dataset = dataset_class(
        data_path=data_path,
        ascii_view=ascii_view,
        max_description_length=max_description_length,
        require_ascii=require_ascii,
        min_num_components=min_num_components,
        max_num_components=max_num_components,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        max_samples=max_samples,
        seed=seed,
        prompt_completion_format=prompt_completion_format,
    )

    # Return the three splits
    return dataset.train, dataset.val, dataset.test
