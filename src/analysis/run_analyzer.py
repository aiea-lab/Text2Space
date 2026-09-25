#!/usr/bin/env python3
"""
Run Analyzer for VLM Evaluation Results

Analyzes evaluation outputs to understand what dataset factors influence model performance.

Usage:
    uv run python src/analysis/run_analyzer.py --run-dir results/qwen3-vl-30b_run005
    uv run python src/analysis/run_analyzer.py --run-dir results/qwen3-vl-30b_run005 --task d
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.constant import ANALYSIS_CONFIG, ANALYSIS_CONFIG_BY_TASK


@dataclass
class FactorGroupMetrics:
    """Metrics for a single factor value group."""

    value: Any
    correct_count: int
    total_count: int
    accuracy: float
    sample_ids: List[str] = field(default_factory=list)
    # Instance-level accuracy (used for Task A and B)
    avg_instance_accuracy: Optional[float] = None
    # Relation-level metrics
    total_correct_relations: Optional[int] = None
    total_relations: Optional[int] = None


@dataclass
class FactorAnalysis:
    """Analysis results for a single factor."""

    factor_name: str
    category: str
    groups: Dict[str, FactorGroupMetrics]
    overall_accuracy: float


@dataclass
class TaskAnalysis:
    """Complete analysis for a single task."""

    task_name: str
    total_samples: int
    overall_accuracy: float
    factor_analyses: Dict[str, FactorAnalysis]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RunAnalyzer:
    """Analyzes VLM evaluation results by dataset factors."""

    DEFAULT_DATASET_PATH = "datasets/spatial_data.jsonl"
    SUPPORTED_TASKS = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # Task E has two separate modes that should be analyzed independently
    TASK_E_MODES = ["e_answer_first", "e_ascii_first"]
    # Tasks with spatial direction answers (includes Task E modes)
    DIRECTION_TASKS = ["c", "d", "e_answer_first", "e_ascii_first", "f", "g", "h"]

    # Ordered labels for confusion matrix
    DIRECTION_LABELS = [
        "above",
        "below",
        "left",
        "right",
        "upper-left",
        "upper-right",
        "lower-left",
        "lower-right",
        "same level",
        "same column",
    ]

    def __init__(
        self,
        run_dir: str,
        dataset_path: Optional[str] = None,
        eval_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize RunAnalyzer.

        Args:
            run_dir: Path to run results folder (e.g., results/qwen3-vl-30b_run005)
            dataset_path: Optional override for dataset JSONL path
            eval_dir: Optional evaluation directory (default: auto-detect)
            output_dir: Optional output directory (default: run_dir/outputs/analysis)

        Raises:
            FileNotFoundError: If run_dir or evaluation folder doesn't exist
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Set eval_dir (configurable or auto-detect)
        if eval_dir:
            self.eval_dir = Path(eval_dir)
        else:
            self.eval_dir = self._find_eval_dir()

        # Set output_dir (configurable or default to outputs/analysis)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.run_dir / "outputs" / "analysis"

        self.run_config = self._load_run_config()
        self.dataset_path = dataset_path or self._get_dataset_path()
        self.dataset = self._load_dataset()
        self.results: Dict[str, TaskAnalysis] = {}

    def _find_eval_dir(self) -> Path:
        """Auto-detect evaluation directory (new or legacy structure)."""
        # Try new structure first: outputs/evaluation/
        new_path = self.run_dir / "outputs" / "evaluation"
        if new_path.exists():
            return new_path

        # Try legacy structure: evaluation/
        legacy_path = self.run_dir / "evaluation"
        if legacy_path.exists():
            return legacy_path

        # Check if evaluation files exist directly in run_dir
        if any(
            (self.run_dir / f"task_{t}_evaluation.json").exists()
            for t in self.SUPPORTED_TASKS
        ):
            return self.run_dir

        raise FileNotFoundError(
            f"No evaluation files found in {self.run_dir}, "
            f"{self.run_dir / 'outputs' / 'evaluation'}, or {self.run_dir / 'evaluation'}"
        )

    def _load_run_config(self) -> dict:
        """Load and return run_config.json from run_dir."""
        config_path = self.run_dir / "run_config.json"
        if not config_path.exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_dataset_path(self) -> str:
        """Resolve dataset path from run_config or use default."""
        config_dataset = self.run_config.get("args", {}).get("dataset")
        if config_dataset:
            return config_dataset
        return self.DEFAULT_DATASET_PATH

    def _load_dataset(self) -> Dict[str, Dict]:
        """Load dataset JSONL and index by 'id' field."""
        dataset = {}
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    dataset[record["id"]] = record
        return dataset

    def _detect_available_tasks(self) -> List[str]:
        """Detect which tasks have evaluation files.

        Supports both exact filenames (task_a_evaluation.json) and
        path-suffixed filenames (task_a_A_grid_1000_evaluation.json).

        Task E is handled specially: instead of merging modes, we detect
        e_answer_first and e_ascii_first as separate tasks.
        """
        tasks = set()
        for task in self.SUPPORTED_TASKS:
            # Task E: detect each mode separately
            if task == "e":
                for mode in ["answer_first", "ascii_first"]:
                    pattern = f"task_e_*{mode}*_evaluation.json"
                    if list(self.eval_dir.glob(pattern)):
                        tasks.add(f"e_{mode}")
                continue

            # Check exact filename first
            exact_file = self.eval_dir / f"task_{task}_evaluation.json"
            if exact_file.exists():
                tasks.add(task)
                continue

            # Check for path-suffixed filenames (e.g., task_a_*_evaluation.json)
            pattern = f"task_{task}_*_evaluation.json"
            if list(self.eval_dir.glob(pattern)):
                tasks.add(task)

        return sorted(tasks)

    def _load_task_evaluation(self, task_name: str) -> Optional[Dict]:
        """Load task evaluation JSON from evaluation subfolder.

        Supports both exact filenames and path-suffixed filenames.
        If multiple files exist for a task, merges their individual_results.

        Task E modes (e_answer_first, e_ascii_first) are handled specially
        to load only the specific mode's file.
        """
        # Handle Task E modes specially
        if task_name in self.TASK_E_MODES:
            # e_answer_first -> answer_first, e_ascii_first -> ascii_first
            mode = task_name.replace("e_", "")
            pattern = f"task_e_*{mode}*_evaluation.json"
            matching_files = list(self.eval_dir.glob(pattern))
            if matching_files:
                with open(matching_files[0], "r", encoding="utf-8") as f:
                    return json.load(f)
            return None

        # Try exact filename first
        exact_file = self.eval_dir / f"task_{task_name}_evaluation.json"
        if exact_file.exists():
            with open(exact_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Check for path-suffixed filenames
        pattern = f"task_{task_name}_*_evaluation.json"
        matching_files = sorted(self.eval_dir.glob(pattern))

        if not matching_files:
            return None

        # If single file, just return it
        if len(matching_files) == 1:
            with open(matching_files[0], "r", encoding="utf-8") as f:
                return json.load(f)

        # Multiple files: merge individual_results
        merged = None
        all_results = []

        for file_path in matching_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if merged is None:
                    merged = data
                all_results.extend(data.get("individual_results", []))

        if merged:
            merged["individual_results"] = all_results
            # Update summary to reflect merged data
            merged["summary"]["total_samples"] = len(all_results)

        return merged

    def _get_factors_for_task(self, task_name: str) -> Dict[str, List[str]]:
        """
        Get all factors to analyze for a given task, organized by category.

        Returns:
            Dict mapping category -> list of factor field names
        """
        task_upper = task_name.upper()
        categories = ANALYSIS_CONFIG_BY_TASK.get(task_upper, [])

        result = {}
        for category in categories:
            fields = ANALYSIS_CONFIG.get(category, [])
            if fields:
                result[category] = fields
        return result

    # Tasks that use instance accuracy (per-sample accuracy) instead of binary correct/incorrect
    INSTANCE_ACCURACY_TASKS = ["a", "b"]

    def _analyze_factor(
        self,
        joined_results: List[Dict],
        factor_name: str,
        category: str,
        task_name: str,
    ) -> FactorAnalysis:
        """
        Analyze performance breakdown by a single factor.

        Args:
            joined_results: List of dicts with both evaluation and dataset fields
            factor_name: Dataset field to group by
            category: Category name for metadata
            task_name: Task identifier for special handling (e.g., Task A/B use instance accuracy)
        """
        groups = defaultdict(
            lambda: {
                "correct": 0,
                "total": 0,
                "ids": [],
                "instance_accuracies": [],  # Per-sample accuracy values
                "correct_relations": 0,
                "total_relations": 0,
            }
        )

        use_instance_accuracy = task_name in self.INSTANCE_ACCURACY_TASKS

        for item in joined_results:
            value = item.get("dataset", {}).get(factor_name)
            if value is None:
                continue

            # Convert to string key for JSON compatibility
            key = str(value)
            groups[key]["total"] += 1
            groups[key]["ids"].append(item["test_id"])

            if item["correct"]:
                groups[key]["correct"] += 1

            # For Task A and B, collect instance-level accuracy
            if use_instance_accuracy:
                eval_data = item.get("evaluation", {})
                if "accuracy" in eval_data:
                    groups[key]["instance_accuracies"].append(eval_data["accuracy"])
                if "correct_count" in eval_data:
                    groups[key]["correct_relations"] += eval_data["correct_count"]
                if "total_relations" in eval_data:
                    groups[key]["total_relations"] += eval_data["total_relations"]

        # Build FactorGroupMetrics for each group
        group_metrics = {}
        total_correct = 0
        total_count = 0
        all_instance_accuracies = []

        for value, counts in groups.items():
            # For Task A and B, use average instance accuracy as the primary metric
            if use_instance_accuracy and counts["instance_accuracies"]:
                avg_instance_acc = sum(counts["instance_accuracies"]) / len(
                    counts["instance_accuracies"]
                )
                accuracy = avg_instance_acc
                all_instance_accuracies.extend(counts["instance_accuracies"])
            else:
                accuracy = (
                    counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                )

            metrics = FactorGroupMetrics(
                value=value,
                correct_count=counts["correct"],
                total_count=counts["total"],
                accuracy=accuracy,
                sample_ids=counts["ids"],
            )

            # Add instance-level metrics for Task A and B
            if use_instance_accuracy and counts["instance_accuracies"]:
                metrics.avg_instance_accuracy = sum(
                    counts["instance_accuracies"]
                ) / len(counts["instance_accuracies"])
                metrics.total_correct_relations = counts["correct_relations"]
                metrics.total_relations = counts["total_relations"]

            group_metrics[value] = metrics
            total_correct += counts["correct"]
            total_count += counts["total"]

        # For Task A and B, overall accuracy is average of instance accuracies
        if use_instance_accuracy and all_instance_accuracies:
            overall_accuracy = sum(all_instance_accuracies) / len(
                all_instance_accuracies
            )
        else:
            overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

        return FactorAnalysis(
            factor_name=factor_name,
            category=category,
            groups=group_metrics,
            overall_accuracy=overall_accuracy,
        )

    def analyze_task(self, task_name: str) -> Optional[TaskAnalysis]:
        """Perform full factor analysis for a single task."""
        eval_data = self._load_task_evaluation(task_name)
        if eval_data is None:
            return None

        individual_results = eval_data.get("individual_results", [])

        # Join evaluation results with dataset records
        joined_results = []
        for result in individual_results:
            test_id = result.get("test_id")
            dataset_record = self.dataset.get(test_id)

            if dataset_record is None:
                continue

            # Task E modes use 'answer_correct' at top level instead of evaluation.correct
            if task_name in self.TASK_E_MODES:
                correct = result.get("answer_correct", False)
            else:
                correct = result.get("evaluation", {}).get("correct", False)

            joined_results.append(
                {
                    "test_id": test_id,
                    "correct": correct,
                    "evaluation": result.get("evaluation", {}),
                    "dataset": dataset_record,
                }
            )

        # Get factors to analyze
        factors_by_category = self._get_factors_for_task(task_name)

        # Analyze each factor
        factor_analyses = {}
        for category, factor_names in factors_by_category.items():
            for factor_name in factor_names:
                analysis = self._analyze_factor(
                    joined_results, factor_name, category, task_name
                )
                factor_analyses[factor_name] = analysis

        # Calculate overall accuracy
        total_correct = sum(1 for r in joined_results if r["correct"])
        total_samples = len(joined_results)

        # For Task A and B, use average instance accuracy instead of binary accuracy
        if task_name in self.INSTANCE_ACCURACY_TASKS:
            instance_accuracies = [
                r["evaluation"].get("accuracy", 0.0)
                for r in joined_results
                if "accuracy" in r.get("evaluation", {})
            ]
            if instance_accuracies:
                overall_accuracy = sum(instance_accuracies) / len(instance_accuracies)
            else:
                overall_accuracy = 0.0
        else:
            overall_accuracy = (
                total_correct / total_samples if total_samples > 0 else 0.0
            )

        return TaskAnalysis(
            task_name=task_name,
            total_samples=total_samples,
            overall_accuracy=overall_accuracy,
            factor_analyses=factor_analyses,
            metadata={
                "run_id": self.run_config.get("run_id", "unknown"),
                "model_name": self.run_config.get("model_name", "unknown"),
            },
        )

    def analyze_all(self) -> Dict[str, TaskAnalysis]:
        """Analyze all available tasks."""
        available_tasks = self._detect_available_tasks()

        for task in available_tasks:
            analysis = self.analyze_task(task)
            if analysis is not None:
                self.results[task] = analysis

        return self.results

    def _to_json(self, analyses: Dict[str, TaskAnalysis]) -> Dict:
        """Convert analyses to JSON-serializable dict."""
        result = {
            "run_id": self.run_config.get("run_id", "unknown"),
            "model_name": self.run_config.get("model_name", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
        }

        for task_name, analysis in analyses.items():
            task_data = {
                "task_name": analysis.task_name,
                "total_samples": analysis.total_samples,
                "overall_accuracy": analysis.overall_accuracy,
                "factor_analyses": {},
            }

            for factor_name, factor_analysis in analysis.factor_analyses.items():
                factor_data = {
                    "factor_name": factor_analysis.factor_name,
                    "category": factor_analysis.category,
                    "overall_accuracy": factor_analysis.overall_accuracy,
                    "groups": {},
                }

                for group_key, metrics in factor_analysis.groups.items():
                    group_data = {
                        "value": metrics.value,
                        "correct_count": metrics.correct_count,
                        "total_count": metrics.total_count,
                        "accuracy": metrics.accuracy,
                    }
                    # Add instance-level accuracy metrics (for Task A and B)
                    if metrics.avg_instance_accuracy is not None:
                        group_data["avg_instance_accuracy"] = (
                            metrics.avg_instance_accuracy
                        )
                    if metrics.total_correct_relations is not None:
                        group_data["total_correct_relations"] = (
                            metrics.total_correct_relations
                        )
                    if metrics.total_relations is not None:
                        group_data["total_relations"] = metrics.total_relations

                    factor_data["groups"][group_key] = group_data

                task_data["factor_analyses"][factor_name] = factor_data

            result["tasks"][task_name] = task_data

        return result

    def save_results(
        self, analyses: Dict[str, TaskAnalysis] = None, output_dir: Optional[str] = None
    ):
        """Save analysis results to JSON file."""
        analyses = analyses or self.results
        output_path = Path(output_dir) if output_dir else self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        json_data = self._to_json(analyses)
        output_file = output_path / "factor_analysis.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_file}")

    def plot_results(
        self,
        analyses: Dict[str, TaskAnalysis] = None,
        output_dir: Optional[str] = None,
        include_factor_plots: bool = False,
    ):
        """Generate plots for analysis results.

        Args:
            analyses: Task analyses to plot (default: self.results)
            output_dir: Output directory (default: self.output_dir)
            include_factor_plots: Whether to generate factor bar charts (default: False).
                                  Confusion matrices are always generated for direction tasks.
        """
        analyses = analyses or self.results
        output_path = Path(output_dir) if output_dir else self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        for task_name, analysis in analyses.items():
            # Generate factor bar charts only if requested
            if include_factor_plots:
                for factor_name, factor_analysis in analysis.factor_analyses.items():
                    self._plot_factor(factor_analysis, task_name, output_path)

            # Generate confusion matrix for direction tasks (always)
            if task_name in self.DIRECTION_TASKS:
                self._plot_confusion_matrix(task_name, output_path)

    def _plot_factor(
        self, factor_analysis: FactorAnalysis, task_name: str, output_dir: Path
    ):
        """Generate a single bar chart for one factor."""
        groups = factor_analysis.groups

        # Sort groups: numeric by value, otherwise alphabetically
        try:
            sorted_keys = sorted(groups.keys(), key=lambda x: float(x))
        except ValueError:
            sorted_keys = sorted(groups.keys())

        values = [groups[k].value for k in sorted_keys]
        accuracies = [groups[k].accuracy for k in sorted_keys]
        counts = [groups[k].total_count for k in sorted_keys]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 5))

        # Bar chart
        bars = ax.bar(
            range(len(values)),
            accuracies,
            color="#4C72B0",
            edgecolor="white",
            linewidth=0.5,
        )

        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.annotate(
                f"n={count}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )

        # Styling
        ax.set_xlabel(self._format_label(factor_analysis.factor_name), fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(
            f"Task {task_name.upper()}: Accuracy by {self._format_label(factor_analysis.factor_name)}",
            fontsize=12,
            fontweight="medium",
        )

        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([self._format_value(v) for v in values], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

        # Clean appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="both", length=0)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="#666666")

        plt.tight_layout()

        # Save
        output_file = output_dir / f"{task_name}_{factor_analysis.factor_name}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"Saved: {output_file}")

    def _plot_confusion_matrix(self, task_name: str, output_dir: Path):
        """Generate confusion matrix heatmaps for a direction task.

        Generates:
        - Combined confusion matrix (all query types)
        - Per-query-type confusion matrices (full_spatial, horizontal, vertical)
        """
        # Load evaluation data
        eval_data = self._load_task_evaluation(task_name)
        if eval_data is None:
            return

        confusion_data = eval_data.get("breakdown", {}).get("confusion_matrix", {})
        if not confusion_data:
            print(f"No confusion matrix data for task {task_name.upper()}")
            return

        # Plot combined confusion matrix
        self._plot_single_confusion_matrix(
            confusion_data, task_name, output_dir, suffix="", title_suffix=""
        )

        # Plot per-query-type confusion matrices
        confusion_by_query_type = eval_data.get("breakdown", {}).get(
            "confusion_matrix_by_query_type", {}
        )
        for query_type, qt_confusion_data in confusion_by_query_type.items():
            if qt_confusion_data:
                self._plot_single_confusion_matrix(
                    qt_confusion_data,
                    task_name,
                    output_dir,
                    suffix=f"_{query_type}",
                    title_suffix=f" ({query_type})",
                )

    def _plot_single_confusion_matrix(
        self,
        confusion_data: Dict[str, int],
        task_name: str,
        output_dir: Path,
        suffix: str = "",
        title_suffix: str = "",
    ):
        """Generate a single confusion matrix heatmap.

        Args:
            confusion_data: Dict mapping "expected_X_predicted_Y" -> count
            task_name: Task identifier
            output_dir: Output directory
            suffix: Filename suffix (e.g., "_full_spatial")
            title_suffix: Title suffix (e.g., " (full_spatial)")
        """
        # Use all 10 direction labels for consistent matrix layout
        labels = self.DIRECTION_LABELS
        n_labels = len(labels)
        label_to_idx = {label: i for i, label in enumerate(labels)}

        # Build confusion matrix
        matrix = np.zeros((n_labels, n_labels), dtype=int)
        for key, count in confusion_data.items():
            parts = key.replace("expected_", "").split("_predicted_")
            if len(parts) == 2:
                true_label, pred_label = parts[0], parts[1]
                if true_label in label_to_idx and pred_label in label_to_idx:
                    matrix[label_to_idx[true_label], label_to_idx[pred_label]] = count

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Count", fontsize=10)

        # Set ticks and labels
        ax.set_xticks(range(n_labels))
        ax.set_yticks(range(n_labels))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Add text annotations
        thresh = matrix.max() / 2.0
        for i in range(n_labels):
            for j in range(n_labels):
                value = matrix[i, j]
                if value > 0:
                    color = "white" if value > thresh else "black"
                    ax.text(
                        j,
                        i,
                        str(value),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

        # Labels and title
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title(
            f"Task {task_name.upper()}: Confusion Matrix{title_suffix}",
            fontsize=12,
            fontweight="medium",
        )

        plt.tight_layout()

        # Save
        output_file = output_dir / f"task_{task_name}_confusion_matrix{suffix}.png"
        plt.savefig(output_file, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"Saved: {output_file}")

    def _format_label(self, name: str) -> str:
        """Format factor name for display."""
        return name.replace("_", " ").title()

    def _format_value(self, value: Any) -> str:
        """Format factor value for display."""
        if isinstance(value, bool) or value in ("True", "False", "true", "false"):
            return str(value).capitalize()
        return str(value)

    def print_summary(self, analyses: Dict[str, TaskAnalysis] = None):
        """Print concise summary to console."""
        analyses = analyses or self.results
        if not analyses:
            return

        print("\nFactor analysis complete:")
        for task_name, analysis in sorted(analyses.items()):
            factors = len(analysis.factor_analyses)
            print(f"  Task {task_name.upper()}: {factors} factors analyzed")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze VLM evaluation results by dataset factors"
    )
    parser.add_argument(
        "--run-dir",
        "-r",
        type=str,
        required=True,
        help="Path to run results folder (e.g., results/qwen3-vl-30b_run005)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=None,
        help="Path to dataset JSONL (default: read from run_config or use datasets/spatial_data.jsonl)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        choices=["a", "b", "c", "d", "e", "f", "g", "h", "all"],
        default="all",
        help="Specific task to analyze (default: all available tasks)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: {run_dir}/outputs/analysis/)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument(
        "--with-factor-plots",
        action="store_true",
        help="Generate factor bar chart PNGs (skipped by default, confusion matrices always generated)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress console output summary"
    )

    args = parser.parse_args()

    try:
        analyzer = RunAnalyzer(
            args.run_dir,
            dataset_path=args.dataset,
            output_dir=Path(args.output_dir) if args.output_dir else None,
        )

        if args.task == "all":
            analyses = analyzer.analyze_all()
        else:
            analysis = analyzer.analyze_task(args.task)
            if analysis:
                analyzer.results[args.task] = analysis
            analyses = analyzer.results

        if not analyses:
            print("No evaluation data found to analyze.")
            sys.exit(1)

        analyzer.save_results()

        if not args.no_plot:
            analyzer.plot_results(include_factor_plots=args.with_factor_plots)

        if not args.quiet:
            analyzer.print_summary(analyses)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
