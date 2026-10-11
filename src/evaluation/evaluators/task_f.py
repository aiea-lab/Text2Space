"""Task F Evaluator: Image + Description + Query -> Answer.

Evaluates model-generated direction answers given image, description, and query.
Uses evaluate_label() to check if the model's answer matches ground truth.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluators.base import DirectionTaskEvaluator


class TaskFEvaluator(DirectionTaskEvaluator):
    """Evaluator for Task F: Image + Description + Query -> Direction answer."""

    task_name = "f"

    # Inherits all evaluation logic from DirectionTaskEvaluator
