"""Task A Evaluator: Description -> ASCII.

Evaluates model-generated ASCII grids against ground truth descriptions.
Uses evaluate_description(mode='verify_ascii') to check if the ASCII
correctly represents the spatial relations stated in the description.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.ascii_evaluator import ASCIIEvaluator
from src.evaluation.evaluators.base import BaseTaskEvaluator, TaskEvaluationResult


class TaskAEvaluator(BaseTaskEvaluator):
    """Evaluator for Task A: Description -> ASCII generation."""

    task_name = "a"

    def __init__(self, dataset: Dict[str, Dict]):
        super().__init__(dataset)
        self.evaluator = ASCIIEvaluator()

    def evaluate(self, results: List[Dict]) -> TaskEvaluationResult:
        """Evaluate ASCII generation results.

        Args:
            results: List of result records with model_output containing 'ascii'

        Returns:
            TaskEvaluationResult with relation-level accuracy metrics
        """
        individual_results = []
        correct_count = 0
        total_correct_relations = 0
        total_relations = 0
        accuracy_sum = 0.0
        accuracy_count = 0

        for record in results:
            test_id = record["test_id"]
            model_output = record.get("model_output", {})

            # Handle string model_output
            if isinstance(model_output, str):
                try:
                    import json

                    model_output = json.loads(model_output)
                except Exception:
                    model_output = {}

            gt_record = self._get_dataset_record(test_id)
            if gt_record is None:
                individual_results.append(
                    {
                        "test_id": test_id,
                        "error": f"Ground truth not found for {test_id}",
                    }
                )
                continue

            # Get ground truth description
            gt_description = gt_record.get("description", "")
            if not gt_description:
                individual_results.append(
                    {"test_id": test_id, "error": "No description in ground truth"}
                )
                continue

            # Extract model ASCII (try multiple formats)
            model_ascii = None
            if isinstance(model_output, dict):
                # Try nested format: {"ascii": {"grid": "..."}}
                ascii_data = model_output.get("ascii", {})
                if isinstance(ascii_data, dict):
                    model_ascii = ascii_data.get("grid") or ascii_data.get("simple")
                elif isinstance(ascii_data, str):
                    model_ascii = ascii_data
                # Try direct format: {"grid": "..."}
                if not model_ascii:
                    model_ascii = model_output.get("grid") or model_output.get("simple")

            if not model_ascii:
                individual_results.append(
                    {
                        "test_id": test_id,
                        "error": "Failed to extract model ASCII",
                        "ground_truth_description": gt_description,
                    }
                )
                continue

            # Evaluate: verify ASCII matches stated relations in description
            eval_result = self.evaluator.evaluate_description(
                model_grid=model_ascii, description=gt_description, mode="verify_ascii"
            )

            is_correct = eval_result.get("correct", False)
            if is_correct:
                correct_count += 1

            # Track relation-level metrics
            rel_accuracy = eval_result.get("accuracy", 0.0)
            rel_total = eval_result.get("total_relations", 0)
            rel_correct = eval_result.get("correct_count", 0)

            total_correct_relations += rel_correct
            total_relations += rel_total
            accuracy_sum += rel_accuracy
            accuracy_count += 1

            individual_results.append(
                {
                    "test_id": test_id,
                    "model_ascii": model_ascii,
                    "ground_truth_description": gt_description,
                    "evaluation": {
                        "correct": is_correct,
                        "accuracy": rel_accuracy,
                        "total_relations": rel_total,
                        "correct_count": rel_correct,
                        "incorrect_count": eval_result.get("incorrect_count", 0),
                        "correct_relations": eval_result.get("correct_relations", []),
                        "incorrect_relations": eval_result.get(
                            "incorrect_relations", []
                        ),
                        "missing_objects": eval_result.get("missing_objects", []),
                        "extra_objects": eval_result.get("extra_objects", []),
                        "error": eval_result.get("error"),
                    },
                }
            )

        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0
        avg_relation_accuracy = (
            accuracy_sum / accuracy_count if accuracy_count > 0 else 0.0
        )

        return TaskEvaluationResult(
            task_name=self.task_name,
            total_samples=total_samples,
            correct_count=correct_count,
            accuracy=accuracy,
            individual_results=individual_results,
            breakdown={
                "total_correct_relations": total_correct_relations,
                "total_relations": total_relations,
                "relation_level_accuracy": (
                    total_correct_relations / total_relations
                    if total_relations > 0
                    else 0.0
                ),
            },
            additional_metrics={
                "average_instance_accuracy": avg_relation_accuracy,
            },
            metadata={"evaluation_mode": "verify_ascii"},
        )
