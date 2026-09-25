"""Task H Evaluator: Image + Description + Query -> Answer (same as Task F).

Evaluates model-generated direction answers given image, description, and query.
Uses evaluate_label() to check if the model's answer matches ground truth.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluators.base import DirectionTaskEvaluator


class TaskHEvaluator(DirectionTaskEvaluator):
    """Evaluator for Task H: Image + Description + Query -> Direction answer (same as F)."""

    task_name = "h"

    # Inherits all evaluation logic from DirectionTaskEvaluator
