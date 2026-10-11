"""Task D Evaluator: Image + Query -> Answer.

Evaluates model-generated direction answers given an image and query.
Uses evaluate_label() to check if the model's answer matches ground truth.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluators.base import DirectionTaskEvaluator


class TaskDEvaluator(DirectionTaskEvaluator):
    """Evaluator for Task D: Image + Query -> Direction answer."""

    task_name = "d"

    # Inherits all evaluation logic from DirectionTaskEvaluator
