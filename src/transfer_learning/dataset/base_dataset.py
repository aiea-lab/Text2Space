"""Base dataset class with common functionality for all task datasets.

This module provides the base class that handles common operations like
loading JSONL files, filtering records, and splitting data into train/val/test sets.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset


class BaseDataset:
    """Base dataset class with common data loading and processing functionality.

    Child classes should implement _create_hf_dataset() to format data
    according to their specific task requirements.
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
        """Initialize base dataset with all splits.

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
        self.ascii_view = ascii_view
        self.max_description_length = max_description_length
        self.prompt_completion_format = prompt_completion_format

        # Load and filter data once
        all_records = self._load_jsonl(data_path)
        filtered_records = self._filter_records(
            all_records,
            require_ascii=require_ascii,
            min_num_components=min_num_components,
            max_num_components=max_num_components,
        )

        # Limit samples if specified
        if max_samples is not None and max_samples < len(filtered_records):
            random.seed(seed)
            filtered_records = random.sample(filtered_records, max_samples)

        # Split data into train/val/test
        train_records, val_records, test_records = self._split_data(
            filtered_records,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )

        # Create split datasets using HuggingFace Dataset for SFTTrainer compatibility
        self.train = self._create_hf_dataset(train_records)
        self.val = self._create_hf_dataset(val_records)
        self.test = self._create_hf_dataset(test_records)

        print(f"Loaded {len(self.train)} train samples")
        print(f"Loaded {len(self.val)} val samples")
        print(f"Loaded {len(self.test)} test samples")

    def _create_hf_dataset(self, records: List[Dict[str, Any]]) -> HFDataset:
        """Create HuggingFace Dataset from records for SFTTrainer.

        This method should be implemented by child classes to format
        data according to their specific task requirements.

        Args:
            records: List of data records

        Returns:
            HuggingFace Dataset with chat format messages
        """
        raise NotImplementedError("Child classes must implement _create_hf_dataset()")

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load records from JSONL file."""
        records = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def _filter_records(
        self,
        records: List[Dict[str, Any]],
        require_ascii: bool = True,
        min_num_components: Optional[int] = None,
        max_num_components: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Filter records based on criteria."""
        filtered = []
        for record in records:
            # Check if ASCII exists
            if require_ascii and not record.get("ascii"):
                continue

            # Check number of components
            num_components = record.get("num_components", 0)
            if min_num_components is not None and num_components < min_num_components:
                continue
            if max_num_components is not None and num_components > max_num_components:
                continue

            filtered.append(record)

        return filtered

    def _split_data(
        self,
        records: List[Dict[str, Any]],
        train_split: float,
        val_split: float,
        test_split: float,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train/val/test sets.

        Returns:
            Tuple of (train_records, val_records, test_records)
        """
        # Ensure splits sum to 1.0
        total = train_split + val_split + test_split
        train_split /= total
        val_split /= total
        test_split /= total

        # Shuffle with seed
        random.seed(seed)
        shuffled = records.copy()
        random.shuffle(shuffled)

        # Calculate split indices
        n = len(shuffled)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)

        # Return all three splits
        train_records = shuffled[:train_end]
        val_records = shuffled[train_end:val_end]
        test_records = shuffled[val_end:]

        return train_records, val_records, test_records
