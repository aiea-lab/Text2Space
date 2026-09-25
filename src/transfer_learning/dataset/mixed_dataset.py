"""Mixed dataset: 50/50 combination of Task A and Task B.

This module handles mixed task fine-tuning where the model learns both
Task A (Description -> ASCII) and Task B (ASCII -> Description) simultaneously.
"""

import random
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets

from .task_a_dataset import TaskADataset
from .task_b_dataset import TaskBDataset


class MixedDataset:
    """Dataset container for mixed Task A + Task B training.

    Loads data and creates 50/50 mix of Task A and Task B samples.
    Access splits via .train, .val, and .test properties.
    """

    def __init__(
        self,
        data_path: str,
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
    ):
        """Initialize mixed dataset with Task A and Task B samples.

        Args:
            data_path: Path to JSONL data file
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
        """
        # Create Task A dataset
        task_a_dataset = TaskADataset(
            data_path=data_path,
            ascii_view=ascii_view,
            max_description_length=max_description_length,
            require_ascii=require_ascii,
            min_num_components=min_num_components,
            max_num_components=max_num_components,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            max_samples=max_samples // 2,
            seed=seed,
            prompt_completion_format=prompt_completion_format,
        )

        # Create Task B dataset with same parameters
        task_b_dataset = TaskBDataset(
            data_path=data_path,
            ascii_view=ascii_view,
            max_description_length=max_description_length,
            require_ascii=require_ascii,
            min_num_components=min_num_components,
            max_num_components=max_num_components,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            max_samples=max_samples // 2,
            seed=seed,
            prompt_completion_format=prompt_completion_format,
        )

        # Mix datasets 50/50 for each split
        self.train = self._mix_datasets(
            task_a_dataset.train, task_b_dataset.train, seed=seed
        )
        self.val = self._mix_datasets(
            task_a_dataset.val, task_b_dataset.val, seed=seed + 1
        )
        self.test = self._mix_datasets(
            task_a_dataset.test, task_b_dataset.test, seed=seed + 2
        )

        print(f"Mixed dataset created:")
        print(f"  Train: {len(self.train)} samples (50% Task A, 50% Task B)")
        print(f"  Val: {len(self.val)} samples (50% Task A, 50% Task B)")
        print(f"  Test: {len(self.test)} samples (50% Task A, 50% Task B)")

    def _mix_datasets(
        self, dataset_a: HFDataset, dataset_b: HFDataset, seed: int
    ) -> HFDataset:
        """Mix two datasets 50/50.

        Args:
            dataset_a: Task A dataset
            dataset_b: Task B dataset
            seed: Random seed for shuffling

        Returns:
            Mixed HuggingFace Dataset
        """
        # Determine the target size (half of each)
        min_size = min(len(dataset_a), len(dataset_b))

        # Sample equal amounts from each dataset
        random.seed(seed)
        indices_a = random.sample(range(len(dataset_a)), min_size)
        indices_b = random.sample(range(len(dataset_b)), min_size)

        # Select subsets
        subset_a = dataset_a.select(indices_a)
        subset_b = dataset_b.select(indices_b)

        # Concatenate
        mixed = concatenate_datasets([subset_a, subset_b])

        # Shuffle the mixed dataset
        mixed = mixed.shuffle(seed=seed)

        return mixed
