"""Task C Evaluator: Description + Query -> Answer.

Evaluates model-generated direction answers given a description and query.
Uses evaluate_label() to check if the model's answer matches ground truth.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluators.base import DirectionTaskEvaluator


class TaskCEvaluator(DirectionTaskEvaluator):
    """Evaluator for Task C: Description + Query -> Direction answer."""

    task_name = "c"

    # Inherits all evaluation logic from DirectionTaskEvaluator
