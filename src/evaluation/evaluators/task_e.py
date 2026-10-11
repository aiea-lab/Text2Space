"""Task E Evaluator: ASCII + Answer Consistency.

Evaluates model-generated ASCII grids and answers for consistency.
Uses:
- evaluate_query() to check if model's ASCII produces its own answer
- evaluate_label() to check if answer matches ground truth
- evaluate_description() for ASCII quality scoring
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluators.base import ConsistencyTaskEvaluator


class TaskEEvaluator(ConsistencyTaskEvaluator):
    """Evaluator for Task E: ASCII-Answer consistency check."""

    task_name = "e"

    # Inherits all evaluation logic from ConsistencyTaskEvaluator
