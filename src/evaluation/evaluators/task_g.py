"""Task G Evaluator: ASCII + Answer Consistency (same as Task E).

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


class TaskGEvaluator(ConsistencyTaskEvaluator):
    """Evaluator for Task G: ASCII-Answer consistency check (same as E)."""

    task_name = "g"

    # Inherits all evaluation logic from ConsistencyTaskEvaluator
