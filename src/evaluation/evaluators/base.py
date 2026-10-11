"""Base class for task evaluators."""

import json
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def compute_classification_metrics(
    confusion_matrix: Dict[str, int],
) -> Dict[str, Any]:
    """Compute precision, recall, F1 from confusion matrix.

    Args:
        confusion_matrix: Dict with keys like "expected_{label}_predicted_{label}"

    Returns:
        Dict containing per-class metrics and aggregate scores
    """
    # Parse confusion matrix to extract labels and counts
    # Format: "expected_{gt}_predicted_{pred}"
    pattern = re.compile(r"expected_(.+)_predicted_(.+)")

    # Collect all unique labels and build count matrix
    labels = set()
    counts = defaultdict(lambda: defaultdict(int))  # counts[gt][pred] = count

    for key, count in confusion_matrix.items():
        match = pattern.match(key)
        if match:
            gt_label, pred_label = match.groups()
            labels.add(gt_label)
            labels.add(pred_label)
            counts[gt_label][pred_label] = count

    if not labels:
        return {
            "per_class": {},
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    labels = sorted(labels)

    # Compute per-class metrics
    per_class = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_support = 0

    for label in labels:
        # True Positives: predicted as label AND actually label
        tp = counts[label][label]

        # False Positives: predicted as label BUT actually something else
        fp = sum(counts[other][label] for other in labels if other != label)

        # False Negatives: actually label BUT predicted as something else
        fn = sum(counts[label][other] for other in labels if other != label)

        # Support: total actual instances of this label
        support = tp + fn

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1: harmonic mean of precision and recall
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_support += support

    # Macro averages (unweighted mean across classes WITH support > 0)
    # Only include classes that actually appear in the data
    classes_with_support = [m for m in per_class.values() if m["support"] > 0]
    n_classes_with_support = len(classes_with_support)

    if n_classes_with_support > 0:
        macro_precision = (
            sum(m["precision"] for m in classes_with_support) / n_classes_with_support
        )
        macro_recall = (
            sum(m["recall"] for m in classes_with_support) / n_classes_with_support
        )
        macro_f1 = sum(m["f1"] for m in classes_with_support) / n_classes_with_support
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    # Micro averages (global TP/FP/FN)
    micro_precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    micro_recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    # Weighted F1 (weighted by support)
    weighted_f1 = (
        sum(m["f1"] * m["support"] for m in per_class.values()) / total_support
        if total_support > 0
        else 0.0
    )

    return {
        "per_class": per_class,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }


@dataclass
class TaskEvaluationResult:
    """Unified container for task evaluation results."""

    task_name: str
    total_samples: int
    correct_count: int
    accuracy: float
    individual_results: List[Dict]
    breakdown: Dict[str, Any] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTaskEvaluator(ABC):
    """Abstract base class for all task evaluators.

    Provides common functionality for loading results, saving outputs,
    and converting to JSON format. Subclasses implement task-specific
    evaluation logic.
    """

    task_name: str = ""  # Override in subclasses: 'a', 'b', 'c', etc.

    def __init__(self, dataset: Dict[str, Dict]):
        """Initialize evaluator with dataset.

        Args:
            dataset: Dict mapping test_id -> dataset record
        """
        self.dataset = dataset

    @abstractmethod
    def evaluate(self, results: List[Dict]) -> TaskEvaluationResult:
        """Evaluate task results.

        Args:
            results: List of result records from JSONL file

        Returns:
            TaskEvaluationResult with evaluation metrics
        """
        pass

    def _extract_model_output(self, record: Dict, key: str) -> Optional[str]:
        """Safely extract model output from record.

        Args:
            record: Task result record
            key: Key to extract (e.g., 'description', 'answer')

        Returns:
            Extracted string value or None if extraction fails
        """
        try:
            output = record.get("model_output", {})
            if isinstance(output, str):
                output = json.loads(output)
            return output.get(key, "")
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None

    def _get_dataset_record(self, test_id: str) -> Optional[Dict]:
        """Get dataset record for a test ID."""
        return self.dataset.get(test_id)

    def to_json(self, result: TaskEvaluationResult, run_id: str = "unknown") -> Dict:
        """Convert evaluation result to JSON-serializable dict.

        Args:
            result: TaskEvaluationResult to convert
            run_id: Run identifier for metadata

        Returns:
            JSON-serializable dictionary
        """
        return {
            "task": result.task_name,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_samples": result.total_samples,
                "correct_count": result.correct_count,
                "accuracy": result.accuracy,
                **result.additional_metrics,
            },
            "breakdown": result.breakdown,
            "individual_results": result.individual_results,
            "metadata": result.metadata,
        }

    def save_results(
        self,
        result: TaskEvaluationResult,
        output_dir: Path,
        run_id: str = "unknown",
    ) -> Path:
        """Save evaluation results to JSON file.

        Args:
            result: TaskEvaluationResult to save
            output_dir: Directory to save to
            run_id: Run identifier for metadata

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_data = self.to_json(result, run_id)
        output_file = output_dir / f"task_{result.task_name}_evaluation.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        return output_file


class DirectionTaskEvaluator(BaseTaskEvaluator):
    """Base class for tasks that evaluate direction answers (C, D, F, H).

    Provides common logic for:
    - Label evaluation with alias support
    - Breakdown by direction and query type
    - Confusion matrix generation
    """

    def __init__(self, dataset: Dict[str, Dict]):
        super().__init__(dataset)
        from src.evaluation.ascii_evaluator import ASCIIEvaluator

        self.evaluator = ASCIIEvaluator()

    def evaluate(self, results: List[Dict]) -> TaskEvaluationResult:
        """Evaluate direction prediction results.

        Args:
            results: List of result records with model_output containing 'answer'

        Returns:
            TaskEvaluationResult with accuracy breakdown by direction and query type
        """
        individual_results = []
        correct_count = 0

        by_direction = defaultdict(lambda: {"correct": 0, "total": 0})
        by_query_type = defaultdict(lambda: {"correct": 0, "total": 0})
        confusion_pairs = defaultdict(int)
        confusion_by_query_type = defaultdict(lambda: defaultdict(int))

        for record in results:
            test_id = record["test_id"]
            model_answer = self._extract_model_output(record, "answer")

            gt_record = self._get_dataset_record(test_id)
            if gt_record is None:
                individual_results.append(
                    {
                        "test_id": test_id,
                        "error": f"Ground truth not found for {test_id}",
                    }
                )
                continue

            gt_label = gt_record["label"]
            query_type = gt_record.get("query_type", "full")
            query_relation = gt_record.get("query_relation", "")

            if model_answer is None:
                individual_results.append(
                    {
                        "test_id": test_id,
                        "error": "Failed to extract model answer",
                        "ground_truth_label": gt_label,
                        "query_type": query_type,
                        "query_relation": query_relation,
                    }
                )
                by_direction[gt_label]["total"] += 1
                by_query_type[query_type]["total"] += 1
                continue

            eval_result = self.evaluator.evaluate_label(model_answer, gt_label)

            # Track all predictions in confusion matrix (including correct ones)
            confusion_key = (
                f"expected_{gt_label}_predicted_{eval_result['predicted_label']}"
            )
            confusion_pairs[confusion_key] += 1
            confusion_by_query_type[query_type][confusion_key] += 1

            if eval_result["correct"]:
                correct_count += 1
                by_direction[gt_label]["correct"] += 1
                by_query_type[query_type]["correct"] += 1

            by_direction[gt_label]["total"] += 1
            by_query_type[query_type]["total"] += 1

            individual_results.append(
                {
                    "test_id": test_id,
                    "model_answer": model_answer,
                    "ground_truth_label": gt_label,
                    "query_type": query_type,
                    "query_relation": query_relation,
                    "evaluation": eval_result,
                }
            )

        total_samples = len(results)
        accuracy = correct_count / total_samples if total_samples > 0 else 0.0

        # Compute F1 metrics from confusion matrix
        classification_metrics = compute_classification_metrics(dict(confusion_pairs))

        # Compute F1 metrics per query type
        f1_by_query_type = {}
        for qt, pairs in confusion_by_query_type.items():
            f1_by_query_type[qt] = compute_classification_metrics(dict(pairs))

        return TaskEvaluationResult(
            task_name=self.task_name,
            total_samples=total_samples,
            correct_count=correct_count,
            accuracy=accuracy,
            individual_results=individual_results,
            breakdown={
                "by_direction": dict(by_direction),
                "by_query_type": dict(by_query_type),
                "confusion_matrix": dict(confusion_pairs),
                "confusion_matrix_by_query_type": {
                    qt: dict(pairs) for qt, pairs in confusion_by_query_type.items()
                },
                "classification_metrics": classification_metrics,
                "classification_metrics_by_query_type": f1_by_query_type,
            },
            additional_metrics={
                "macro_f1": classification_metrics["macro_f1"],
                "micro_f1": classification_metrics["micro_f1"],
                "weighted_f1": classification_metrics["weighted_f1"],
            },
        )


class ConsistencyTaskEvaluator(BaseTaskEvaluator):
    """Base class for tasks that evaluate ASCII-answer consistency (E).

    Provides common logic for:
    - ASCII-answer consistency check
    - Answer correctness evaluation
    - ASCII quality scoring
    - Bucket analysis by ASCII accuracy
    """

    def __init__(self, dataset: Dict[str, Dict]):
        super().__init__(dataset)
        from src.evaluation.ascii_evaluator import ASCIIEvaluator

        self.evaluator = ASCIIEvaluator()

    @staticmethod
    def _get_accuracy_bucket(accuracy: float) -> str:
        """Categorize accuracy into buckets."""
        if accuracy < 0.25:
            return "[0, 0.25)"
        elif accuracy < 0.5:
            return "[0.25, 0.5)"
        elif accuracy < 0.75:
            return "[0.5, 0.75)"
        else:
            return "[0.75, 1]"

    def evaluate(self, results: List[Dict]) -> TaskEvaluationResult:
        """Evaluate ASCII-answer consistency results.

        Args:
            results: List of result records with model_output containing 'answer' and 'ascii'

        Returns:
            TaskEvaluationResult with consistency and accuracy metrics
        """
        individual_results = []
        stats = defaultdict(int)
        confusion_pairs = defaultdict(int)
        confusion_by_query_type = defaultdict(lambda: defaultdict(int))

        # Accuracy bucket tracking
        accuracy_buckets = {
            bucket: {
                "consistent": 0,
                "inconsistent": 0,
                "answer_correct": 0,
                "total": 0,
                "ascii_accuracy_sum": 0.0,
                # Dataset attribute sums for averages
                "num_components_sum": 0,
                "num_relations_sum": 0,
                "ambiguous_stages_sum": 0,
            }
            for bucket in ["[0, 0.25)", "[0.25, 0.5)", "[0.5, 0.75)", "[0.75, 1]"]
        }

        # Error mapping taxonomy: ASCII correctness × Answer correctness
        error_mapping = {
            "correct_ascii_correct_answer": 0,
            "correct_ascii_wrong_answer": 0,
            "wrong_ascii_correct_answer": 0,
            "wrong_ascii_wrong_answer": 0,
        }

        total_ascii_accuracy_sum = 0.0
        total_ascii_accuracy_count = 0

        for record in results:
            test_id = record["test_id"]
            model_output = record.get("model_output", {})

            # Handle string model_output
            if isinstance(model_output, str):
                try:
                    model_output = json.loads(model_output)
                except json.JSONDecodeError:
                    model_output = {}

            # Handle string turn1_output
            turn1_output = record.get("turn1_output", {})
            if isinstance(turn1_output, str):
                try:
                    turn1_output = json.loads(turn1_output)
                except json.JSONDecodeError:
                    turn1_output = {}

            # Skip if model had an error
            if "error" in model_output:
                stats["skipped_error"] += 1
                individual_results.append(
                    {
                        "test_id": test_id,
                        "status": "skipped",
                        "reason": model_output.get("error", "model error"),
                    }
                )
                continue

            # Extract answer from model_output or turn1_output (answer_first workflow)
            model_answer = model_output.get("answer") or turn1_output.get("answer")
            if not model_answer:
                stats["skipped_error"] += 1
                individual_results.append(
                    {
                        "test_id": test_id,
                        "status": "skipped",
                        "reason": "missing answer",
                    }
                )
                continue

            # Get dataset entry for query info
            gt_record = self._get_dataset_record(test_id)
            if gt_record is None:
                stats["skipped_not_found"] += 1
                individual_results.append(
                    {
                        "test_id": test_id,
                        "status": "skipped",
                        "reason": "test_id not found in dataset",
                    }
                )
                continue

            query = gt_record["query_relation"]

            # Extract ASCII from model_output or turn1_output (ascii_first workflow)
            model_ascii = model_output.get("ascii", {}).get(
                "grid", ""
            ) or turn1_output.get("ascii", {}).get("grid", "")

            if not model_ascii:
                stats["skipped_no_ascii"] += 1
                individual_results.append(
                    {
                        "test_id": test_id,
                        "status": "skipped",
                        "reason": "no ASCII grid in model output",
                    }
                )
                continue

            # Evaluate consistency: does the model's ASCII produce the model's answer?
            consistency_result = self.evaluator.evaluate_query(
                model_grid=model_ascii, query=query, reference_label=model_answer
            )

            # Evaluate ASCII accuracy using description-based evaluation
            description = gt_record.get("description", "")
            ascii_accuracy = 0.0

            if description:
                desc_eval = self.evaluator.evaluate_description(
                    model_grid=model_ascii, description=description, mode="verify_ascii"
                )
                if "accuracy" in desc_eval:
                    ascii_accuracy = desc_eval["accuracy"]
                elif desc_eval.get("correct"):
                    ascii_accuracy = 1.0

            ascii_accuracy_bucket = self._get_accuracy_bucket(ascii_accuracy)

            # Evaluate answer accuracy against ground truth label
            ground_truth_label = gt_record.get("label", "")
            label_eval = self.evaluator.evaluate_label(model_answer, ground_truth_label)
            answer_correct = label_eval["correct"]

            # Evaluate ASCII correctness: does ASCII produce ground truth answer?
            ascii_correctness_result = self.evaluator.evaluate_query(
                model_grid=model_ascii, query=query, reference_label=ground_truth_label
            )
            ascii_correct = ascii_correctness_result["correct"]

            # Determine error category (ASCII correctness × Answer correctness)
            if ascii_correct and answer_correct:
                error_category = "correct_ascii_correct_answer"
            elif ascii_correct and not answer_correct:
                error_category = "correct_ascii_wrong_answer"
            elif not ascii_correct and answer_correct:
                error_category = "wrong_ascii_correct_answer"
            else:
                error_category = "wrong_ascii_wrong_answer"
            error_mapping[error_category] += 1

            # Track confusion matrix
            query_type = gt_record.get("query_type", "full")
            confusion_key = f"expected_{ground_truth_label}_predicted_{label_eval['predicted_label']}"
            confusion_pairs[confusion_key] += 1
            confusion_by_query_type[query_type][confusion_key] += 1

            stats["evaluated"] += 1
            if consistency_result["correct"]:
                stats["consistent"] += 1
                accuracy_buckets[ascii_accuracy_bucket]["consistent"] += 1
                # Track answer correctness among consistent instances
                if answer_correct:
                    stats["consistent_and_correct"] += 1
            else:
                stats["inconsistent"] += 1
                accuracy_buckets[ascii_accuracy_bucket]["inconsistent"] += 1

            if answer_correct:
                stats["answer_correct"] += 1
                accuracy_buckets[ascii_accuracy_bucket]["answer_correct"] += 1

            accuracy_buckets[ascii_accuracy_bucket]["total"] += 1
            accuracy_buckets[ascii_accuracy_bucket][
                "ascii_accuracy_sum"
            ] += ascii_accuracy

            # Track dataset attributes for bucket averages
            accuracy_buckets[ascii_accuracy_bucket][
                "num_components_sum"
            ] += gt_record.get("num_components", 0)
            accuracy_buckets[ascii_accuracy_bucket][
                "num_relations_sum"
            ] += gt_record.get("num_relations", 0)
            accuracy_buckets[ascii_accuracy_bucket][
                "ambiguous_stages_sum"
            ] += gt_record.get("ambiguous_stages", 0)

            total_ascii_accuracy_sum += ascii_accuracy
            total_ascii_accuracy_count += 1

            individual_results.append(
                {
                    "test_id": test_id,
                    "status": "evaluated",
                    "consistent": consistency_result["correct"],
                    "answer_correct": answer_correct,
                    "ascii_correct": ascii_correct,
                    "error_category": error_category,
                    "query": query,
                    "query_type": consistency_result.get("query_type"),
                    "model_answer": model_answer,
                    "ground_truth_label": ground_truth_label,
                    "inferred_from_ascii": consistency_result.get("actual"),
                    "ascii_accuracy": ascii_accuracy,
                    "ascii_accuracy_bucket": ascii_accuracy_bucket,
                    "evaluation": label_eval,
                    "error": consistency_result.get("error"),
                }
            )

        # Calculate metrics
        total_samples = len(results)
        evaluated_count = stats["evaluated"]
        consistency_rate = (
            stats["consistent"] / evaluated_count if evaluated_count > 0 else 0.0
        )
        answer_accuracy = (
            stats["answer_correct"] / evaluated_count if evaluated_count > 0 else 0.0
        )
        avg_ascii_accuracy = (
            total_ascii_accuracy_sum / total_ascii_accuracy_count
            if total_ascii_accuracy_count > 0
            else 0.0
        )
        # Accuracy among consistent instances
        consistent_accuracy = (
            stats["consistent_and_correct"] / stats["consistent"]
            if stats["consistent"] > 0
            else 0.0
        )

        # Calculate bucket stats
        bucket_analysis = {}
        for bucket, counts in accuracy_buckets.items():
            bucket_total = counts["total"]
            if bucket_total > 0:
                bucket_analysis[bucket] = {
                    "total": bucket_total,
                    "consistent": counts["consistent"],
                    "inconsistent": counts["inconsistent"],
                    "consistency_rate": counts["consistent"] / bucket_total,
                    "answer_correct": counts["answer_correct"],
                    "answer_accuracy": counts["answer_correct"] / bucket_total,
                    "avg_ascii_accuracy": counts["ascii_accuracy_sum"] / bucket_total,
                    # Dataset attribute averages
                    "avg_num_components": counts["num_components_sum"] / bucket_total,
                    "avg_num_relations": counts["num_relations_sum"] / bucket_total,
                    "avg_ambiguous_stages": counts["ambiguous_stages_sum"]
                    / bucket_total,
                }
            else:
                bucket_analysis[bucket] = {
                    "total": 0,
                    "consistent": 0,
                    "inconsistent": 0,
                    "consistency_rate": 0.0,
                    "answer_correct": 0,
                    "answer_accuracy": 0.0,
                    "avg_ascii_accuracy": 0.0,
                    "avg_num_components": 0.0,
                    "avg_num_relations": 0.0,
                    "avg_ambiguous_stages": 0.0,
                }

        # Calculate ASCII correctness rate
        ascii_correct_count = (
            error_mapping["correct_ascii_correct_answer"]
            + error_mapping["correct_ascii_wrong_answer"]
        )
        ascii_correctness_rate = (
            ascii_correct_count / evaluated_count if evaluated_count > 0 else 0.0
        )

        # Compute F1 metrics from confusion matrix
        classification_metrics = compute_classification_metrics(dict(confusion_pairs))

        # Compute F1 metrics per query type
        f1_by_query_type = {}
        for qt, pairs in confusion_by_query_type.items():
            f1_by_query_type[qt] = compute_classification_metrics(dict(pairs))

        return TaskEvaluationResult(
            task_name=self.task_name,
            total_samples=total_samples,
            correct_count=stats["answer_correct"],
            accuracy=answer_accuracy,
            individual_results=individual_results,
            breakdown={
                "by_ascii_accuracy": bucket_analysis,
                "skip_reasons": {
                    "model_error": stats["skipped_error"],
                    "not_in_dataset": stats["skipped_not_found"],
                    "no_ascii": stats["skipped_no_ascii"],
                },
                "confusion_matrix": dict(confusion_pairs),
                "confusion_matrix_by_query_type": {
                    qt: dict(pairs) for qt, pairs in confusion_by_query_type.items()
                },
                "error_mapping": error_mapping,
                "classification_metrics": classification_metrics,
                "classification_metrics_by_query_type": f1_by_query_type,
            },
            additional_metrics={
                "evaluated": evaluated_count,
                "consistent": stats["consistent"],
                "inconsistent": stats["inconsistent"],
                "consistency_rate": consistency_rate,
                "consistent_and_correct": stats["consistent_and_correct"],
                "consistent_accuracy": consistent_accuracy,
                "avg_ascii_accuracy": avg_ascii_accuracy,
                "ascii_correct_count": ascii_correct_count,
                "ascii_correctness_rate": ascii_correctness_rate,
                "macro_f1": classification_metrics["macro_f1"],
                "micro_f1": classification_metrics["micro_f1"],
                "weighted_f1": classification_metrics["weighted_f1"],
            },
        )
