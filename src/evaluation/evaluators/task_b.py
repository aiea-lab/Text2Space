"""Task B Evaluator: Image -> Description.

Evaluates model-generated descriptions against ground truth ASCII grids.
Uses evaluate_description(mode='verify_description') to check if the model's
description correctly covers the spatial relations in the ground truth ASCII.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.ascii_evaluator import ASCIIEvaluator
from src.evaluation.evaluators.base import BaseTaskEvaluator, TaskEvaluationResult


class TaskBEvaluator(BaseTaskEvaluator):
    """Evaluator for Task B: Image -> Description generation."""

    task_name = "b"

    def __init__(self, dataset: Dict[str, Dict]):
        super().__init__(dataset)
        self.evaluator = ASCIIEvaluator()

    def evaluate(self, results: List[Dict]) -> TaskEvaluationResult:
        """Evaluate description generation results.

        Args:
            results: List of result records with model_output containing 'description'

        Returns:
            TaskEvaluationResult with relation-level accuracy metrics
        """
        individual_results = []
        correct_count = 0
        total_correct_relations = 0
        total_relations = 0
        total_incorrect = 0
        accuracy_sum = 0.0
        accuracy_count = 0

        for record in results:
            test_id = record["test_id"]
            model_desc = self._extract_model_output(record, "description")

            gt_record = self._get_dataset_record(test_id)
            if gt_record is None:
                individual_results.append(
                    {
                        "test_id": test_id,
                        "error": f"Ground truth not found for {test_id}",
                    }
                )
                continue

            # Get ground truth ASCII
            gt_ascii = gt_record.get("ascii", {}).get("simple", "")
            if not gt_ascii:
                gt_ascii = gt_record.get("ascii", {}).get("grid", "")

            if not gt_ascii:
                individual_results.append(
                    {"test_id": test_id, "error": "No ASCII in ground truth"}
                )
                continue

            if model_desc is None:
                individual_results.append(
                    {
                        "test_id": test_id,
                        "error": "Failed to extract model description",
                        "ground_truth_ascii": gt_ascii,
                    }
                )
                continue

            # Evaluate: verify description covers spatial relations in ASCII
            eval_result = self.evaluator.evaluate_description(
                model_grid=gt_ascii, description=model_desc, mode="verify_description"
            )

            is_correct = eval_result.get("correct", False)
            if is_correct:
                correct_count += 1

            # Track relation-level metrics
            total_incorrect += eval_result.get("incorrect_count", 0)
            total_correct_relations += eval_result.get("correct_count", 0)
            total_relations += eval_result.get("total_relations", 0)
            accuracy_sum += eval_result.get("accuracy", 0.0)
            accuracy_count += 1

            individual_result = {
                "test_id": test_id,
                "model_description": model_desc,
                "ground_truth_ascii": gt_ascii,
                "evaluation": {
                    "correct": is_correct,
                    "accuracy": eval_result.get("accuracy", 0.0),
                    "total_relations": eval_result.get("total_relations", 0),
                    "correct_count": eval_result.get("correct_count", 0),
                    "incorrect_count": eval_result.get("incorrect_count", 0),
                    "correct_relations": eval_result.get("correct_relations", []),
                    "incorrect_relations": eval_result.get("incorrect_relations", []),
                },
            }

            # Add objects_not_in_grid if present
            if eval_result.get("objects_not_in_grid"):
                individual_result["evaluation"]["objects_not_in_grid"] = eval_result[
                    "objects_not_in_grid"
                ]

            individual_results.append(individual_result)

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
                "incorrect_relations_total": total_incorrect,
            },
            additional_metrics={
                "average_instance_accuracy": avg_relation_accuracy,
            },
            metadata={"evaluation_mode": "verify_description"},
        )
