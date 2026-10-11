"""Task A dataset: Description -> ASCII generation.

This module handles Task A fine-tuning where the model learns to generate
ASCII diagrams from textual descriptions.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset as HFDataset

from .base_dataset import BaseDataset

# Add experiment/inference to path to import llm_helpers
project_root = Path(__file__).resolve().parents[3]
inference_path = project_root / "src" / "experiment" / "inference"
if str(inference_path) not in sys.path:
    sys.path.insert(0, str(inference_path))

from src.experiment.inference.llm_helpers import build_messages


class TaskADataset(BaseDataset):
    """Dataset container for Task A: Description -> ASCII generation.

    Loads data from JSONL format once and creates train/val/test splits internally.
    Access splits via .train, .val, and .test properties.
    """

    def _create_hf_dataset(self, records: List[Dict[str, Any]]) -> HFDataset:
        """Create HuggingFace Dataset from records for Task A (Description -> ASCII).

        Args:
            records: List of data records

        Returns:
            HuggingFace Dataset with chat format messages or prompt/completion format
        """
        # Initialize formatted_data based on format type
        if self.prompt_completion_format:
            formatted_data = {
                "prompt": [],
                "completion": [],
                "description": [],
                "ascii": [],
                "record_id": [],
                "num_components": [],
            }
        else:
            formatted_data = {
                "messages": [],
                "description": [],
                "ascii": [],
                "record_id": [],
                "num_components": [],
            }

        for idx, record in enumerate(records):
            # Extract description and ASCII
            description = record.get("description", "")
            ascii_data = record.get("ascii", {})
            if isinstance(ascii_data, dict):
                ascii_output = ascii_data.get(self.ascii_view, "")
            else:
                ascii_output = str(ascii_data)

            # Format output
            output_text = ascii_output.strip()

            # Create messages using build_messages for Task A
            messages = build_messages(
                question="",
                description=description,
                few_shot_examples=None,
                ascii_format_input=None,
                ascii_format_output=self.ascii_view,
                prompt_mode="simple",
                few_shot_mode="system",
                ascii_order="answer_first",
                ascii_input=None,
                task="A",
            )

            # Add to formatted data based on format type
            if self.prompt_completion_format:
                formatted_data["prompt"].append(messages)
                formatted_data["completion"].append(
                    [{"role": "assistant", "content": output_text}]
                )
            else:
                # Add assistant response
                messages.append({"role": "assistant", "content": output_text})

                formatted_data["messages"].append(messages)

            formatted_data["description"].append(description)
            formatted_data["ascii"].append(ascii_output)
            formatted_data["record_id"].append(record.get("id", f"sample_{idx}"))
            formatted_data["num_components"].append(record.get("num_components", 0))

        # Create HuggingFace Dataset
        return HFDataset.from_dict(formatted_data)
