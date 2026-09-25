"""Task-specific evaluators for VLM inference results."""

import sys
from pathlib import Path

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.evaluators.base import BaseTaskEvaluator, TaskEvaluationResult
from src.evaluation.evaluators.task_a import TaskAEvaluator
from src.evaluation.evaluators.task_b import TaskBEvaluator
from src.evaluation.evaluators.task_c import TaskCEvaluator
from src.evaluation.evaluators.task_d import TaskDEvaluator
from src.evaluation.evaluators.task_e import TaskEEvaluator
from src.evaluation.evaluators.task_f import TaskFEvaluator
from src.evaluation.evaluators.task_g import TaskGEvaluator
from src.evaluation.evaluators.task_h import TaskHEvaluator

__all__ = [
    "BaseTaskEvaluator",
    "TaskEvaluationResult",
    "TaskAEvaluator",
    "TaskBEvaluator",
    "TaskCEvaluator",
    "TaskDEvaluator",
    "TaskEEvaluator",
    "TaskFEvaluator",
    "TaskGEvaluator",
    "TaskHEvaluator",
]

# Task name to evaluator class mapping
TASK_EVALUATORS = {
    "a": TaskAEvaluator,
    "b": TaskBEvaluator,
    "c": TaskCEvaluator,
    "d": TaskDEvaluator,
    "e": TaskEEvaluator,
    "f": TaskFEvaluator,
    "g": TaskGEvaluator,
    "h": TaskHEvaluator,
}


def get_evaluator(task_name: str) -> type:
    """Get evaluator class for a task name."""
    task_lower = task_name.lower()
    if task_lower not in TASK_EVALUATORS:
        raise ValueError(
            f"Unknown task: {task_name}. Valid tasks: {list(TASK_EVALUATORS.keys())}"
        )
    return TASK_EVALUATORS[task_lower]
