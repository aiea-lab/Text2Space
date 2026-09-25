#!/usr/bin/env python3
"""Unified Benchmark Pipeline for VLM Spatial Reasoning Tasks.

Runs both evaluation AND analysis by default. Outputs to:
  results/model/outputs/evaluation/  - Evaluation JSONs
  results/model/outputs/analysis/    - Factor analysis JSON + confusion matrix PNGs

Usage (from project root):
    # Default: run both evaluation AND analysis for a single model
    uv run python src/benchmark.py --results-dir results/llama70BInstruct

    # Benchmark ALL models in a directory
    uv run python src/benchmark.py --results-root results/

    # Evaluate only (skip analysis)
    uv run python src/benchmark.py --results-dir results/llama70BInstruct --evaluate-only

    # Analyze only (skip evaluation, requires existing evaluation JSONs)
    uv run python src/benchmark.py --results-dir results/llama70BInstruct --analyze-only

    # Preview without saving
    uv run python src/benchmark.py --results-dir results/llama70BInstruct --no-save

    # Specific tasks only
    uv run python src/benchmark.py --results-dir results/llama70BInstruct --task d --task e

    # Include factor bar chart PNGs (skipped by default)
    uv run python src/benchmark.py --results-dir results/llama70BInstruct --with-factor-plots
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluators import (
    BaseTaskEvaluator,
    TaskEvaluationResult,
    get_evaluator,
)


class BenchmarkPipeline:
    """Unified benchmark pipeline: evaluation + analysis."""

    # File naming patterns to detect task type
    # Supports: task_X.jsonl, task_X_results.jsonl
    RESULT_FILE_PATTERNS = [
        r"task_([a-h])_results\.jsonl$",  # New format
        r"task_([a-h])\.jsonl$",  # Old format
    ]

    DEFAULT_DATASET_PATH = Path("datasets/spatial_data.jsonl")

    def __init__(
        self,
        results_dir: Path,
        dataset_path: Optional[Path] = None,
    ):
        """Initialize pipeline.

        Args:
            results_dir: Path to directory containing result JSONL files
            dataset_path: Path to dataset JSONL (default: datasets/spatial_data.jsonl)
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        # Output directories
        self.outputs_dir = self.results_dir / "outputs"
        self.eval_dir = self.outputs_dir / "evaluation"
        self.analysis_dir = self.outputs_dir / "analysis"

        self.dataset_path = dataset_path or self._resolve_dataset_path()
        self.dataset = self._load_dataset()
        self.run_config = self._load_run_config()

    def _resolve_dataset_path(self) -> Path:
        """Resolve dataset path from run_config or use default."""
        config_path = self.results_dir / "run_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                config_dataset = config.get("args", {}).get("dataset")
                if config_dataset:
                    return Path(config_dataset)
        return self.DEFAULT_DATASET_PATH

    def _load_dataset(self) -> Dict[str, Dict]:
        """Load dataset JSONL and index by 'id' field."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        dataset = {}
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    dataset[record["id"]] = record
        return dataset

    def _load_run_config(self) -> Dict:
        """Load run_config.json if it exists."""
        config_path = self.results_dir / "run_config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _detect_tasks(self) -> List[Tuple[str, Path]]:
        """Auto-detect available tasks from result files.

        Recursively searches the results directory for task result files.
        Supports directory structures like:
        - results/model/task_d.jsonl (flat)
        - results/model/D/grid/1000/task_d_results.jsonl (nested)

        Returns:
            List of (task_name, file_path) tuples, sorted by task name then path
        """
        tasks = []

        # Recursively find all JSONL files
        for file_path in self.results_dir.rglob("*.jsonl"):
            if not file_path.is_file():
                continue

            for pattern in self.RESULT_FILE_PATTERNS:
                match = re.search(pattern, file_path.name, re.IGNORECASE)
                if match:
                    task_name = match.group(1).lower()
                    tasks.append((task_name, file_path))
                    break

        # Sort by task name, then by path (for consistent ordering)
        return sorted(tasks, key=lambda x: (x[0], str(x[1])))

    def _load_results(self, file_path: Path) -> List[Dict]:
        """Load results from JSONL file."""
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        return results

    def evaluate(
        self, tasks: Optional[List[str]] = None
    ) -> List[Tuple[str, Path, TaskEvaluationResult, BaseTaskEvaluator]]:
        """Run evaluation for specified tasks (auto-detect if None).

        Args:
            tasks: List of task names to evaluate, or None to auto-detect

        Returns:
            List of (task_name, file_path, TaskEvaluationResult, evaluator) tuples
        """
        detected_tasks = self._detect_tasks()

        if tasks:
            # Filter to requested tasks
            task_set = {t.lower() for t in tasks}
            detected_tasks = [(t, p) for t, p in detected_tasks if t in task_set]

        if not detected_tasks:
            raise ValueError(f"No task result files found in {self.results_dir}")

        results = []
        for task_name, file_path in detected_tasks:
            # Show relative path from results_dir
            rel_path = file_path.relative_to(self.results_dir)
            print(f"Evaluating Task {task_name.upper()} from {rel_path}...")

            # Load results
            task_results = self._load_results(file_path)

            # Get evaluator and run evaluation
            evaluator_class = get_evaluator(task_name)
            evaluator = evaluator_class(self.dataset)
            eval_result = evaluator.evaluate(task_results)

            # Store evaluator instance to avoid re-instantiation in save_results()
            results.append((task_name, file_path, eval_result, evaluator))
            print(
                f"  Task {task_name.upper()}: {eval_result.accuracy:.2%} "
                f"({eval_result.correct_count}/{eval_result.total_samples})"
            )

        return results

    def save_results(
        self,
        results: List[Tuple[str, Path, TaskEvaluationResult, BaseTaskEvaluator]],
    ) -> List[Path]:
        """Save evaluation results to JSON files.

        Saves all results to results_dir/outputs/evaluation/ folder.
        For tasks with same name in different paths, adds path suffix to filename.

        Args:
            results: List of (task_name, file_path, result, evaluator) tuples

        Returns:
            List of saved file paths
        """
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        run_id = self.run_config.get("run_id", "unknown")
        saved_files = []

        for task_name, file_path, result, evaluator in results:
            # Create unique filename for duplicate tasks
            rel_path = file_path.relative_to(self.results_dir).parent
            if str(rel_path) != ".":
                # Use path components to create unique suffix
                path_suffix = str(rel_path).replace("/", "_").replace("\\", "_")
                output_filename = f"task_{task_name}_{path_suffix}_evaluation.json"
            else:
                output_filename = f"task_{task_name}_evaluation.json"

            # Save with custom filename (reuse evaluator from evaluate())
            output_file = self.eval_dir / output_filename
            eval_json = evaluator.to_json(result, run_id)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(eval_json, f, indent=2, ensure_ascii=False)

            saved_files.append(output_file)
            print(f"Saved: {output_file}")

        # Save aggregated summary
        summary = self._create_summary(results)
        summary_file = self.eval_dir / "evaluation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        saved_files.append(summary_file)
        print(f"Saved: {summary_file}")

        return saved_files

    def _create_summary(
        self, results: List[Tuple[str, Path, TaskEvaluationResult, BaseTaskEvaluator]]
    ) -> Dict:
        """Create aggregated summary across all tasks."""
        tasks_summary = {}
        for task_name, file_path, result, _evaluator in results:
            # Use relative path as key to distinguish same task in different dirs
            rel_path = str(file_path.relative_to(self.results_dir).parent)
            key = f"{task_name}:{rel_path}" if rel_path != "." else task_name

            tasks_summary[key] = {
                "task": task_name,
                "path": rel_path,
                "total_samples": result.total_samples,
                "correct_count": result.correct_count,
                "accuracy": result.accuracy,
                **result.additional_metrics,
            }

        return {
            "run_id": self.run_config.get("run_id", "unknown"),
            "model_name": self.run_config.get("model_name", "unknown"),
            "results_dir": str(self.results_dir),
            "tasks": tasks_summary,
        }

    def analyze(
        self, tasks: Optional[List[str]] = None, include_factor_plots: bool = False
    ) -> Dict:
        """Run factor analysis using existing RunAnalyzer.

        Args:
            tasks: List of task names to analyze, or None for all
            include_factor_plots: Whether to generate factor bar chart PNGs
                                  (default: False, confusion matrices always generated)

        Returns:
            Analysis results dictionary
        """
        from src.analysis.run_analyzer import RunAnalyzer

        print(f"\nRunning factor analysis...")

        analyzer = RunAnalyzer(
            run_dir=self.results_dir,
            eval_dir=self.eval_dir,
            output_dir=self.analysis_dir,
        )

        if tasks:
            for task in tasks:
                analysis = analyzer.analyze_task(task.lower())
                if analysis:
                    analyzer.results[task.lower()] = analysis
        else:
            analyzer.analyze_all()

        analyzer.save_results()
        analyzer.plot_results(include_factor_plots=include_factor_plots)
        analyzer.print_summary()

        return analyzer.results

    def print_summary(
        self, results: List[Tuple[str, Path, TaskEvaluationResult, BaseTaskEvaluator]]
    ):
        """Print concise summary to console."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        for task_name, file_path, result, _evaluator in results:
            rel_path = file_path.relative_to(self.results_dir).parent
            path_str = f" ({rel_path})" if str(rel_path) != "." else ""

            # Basic accuracy line with evaluated/skipped count for consistency tasks
            evaluated = result.additional_metrics.get("evaluated")
            if evaluated is not None:
                skipped = result.total_samples - evaluated
                line = f"Task {task_name.upper()}{path_str}: {result.accuracy:.2%} ({result.correct_count}/{evaluated}, {skipped} skipped)"
            else:
                line = f"Task {task_name.upper()}{path_str}: {result.accuracy:.2%} ({result.correct_count}/{result.total_samples})"

            # Add instance accuracy for Tasks A/B
            if "average_instance_accuracy" in result.additional_metrics:
                instance_acc = result.additional_metrics["average_instance_accuracy"]
                line += f", instance_acc={instance_acc:.2%}"

            # Add ascii accuracy and consistency rate if available (Task E)
            if "avg_ascii_accuracy" in result.additional_metrics:
                ascii_acc = result.additional_metrics["avg_ascii_accuracy"]
                line += f", ascii_acc={ascii_acc:.2%}"
            if "consistency_rate" in result.additional_metrics:
                consistency = result.additional_metrics["consistency_rate"]
                line += f", consistency={consistency:.2%}"
            if "consistent_accuracy" in result.additional_metrics:
                consistent_acc = result.additional_metrics["consistent_accuracy"]
                line += f", consistent_acc={consistent_acc:.2%}"

            # Add F1 metrics if available
            if "macro_f1" in result.additional_metrics:
                macro_f1 = result.additional_metrics["macro_f1"]
                line += f", macro_f1={macro_f1:.2%}"

            print(line)

        print("=" * 60)

    def run(
        self,
        tasks: Optional[List[str]] = None,
        save: bool = True,
        do_evaluate: bool = True,
        do_analyze: bool = True,
        include_factor_plots: bool = False,
    ) -> Optional[List[Tuple[str, Path, TaskEvaluationResult, BaseTaskEvaluator]]]:
        """Full pipeline: evaluate → analyze (both by default).

        Args:
            tasks: List of task names to evaluate, or None to auto-detect
            save: Whether to save results to files
            do_evaluate: Whether to run evaluation
            do_analyze: Whether to run analysis
            include_factor_plots: Whether to generate factor bar chart PNGs
                                  (default: False, confusion matrices always generated)

        Returns:
            List of (task_name, file_path, TaskEvaluationResult, evaluator) tuples,
            or None if do_evaluate is False
        """
        results = None

        if do_evaluate:
            results = self.evaluate(tasks)

            if save:
                self.save_results(results)

            self.print_summary(results)

        if do_analyze and save:  # Analysis requires saved evaluation files
            self.analyze(tasks, include_factor_plots=include_factor_plots)

        return results


def _discover_model_dirs(results_root: Path) -> List[Path]:
    """Discover model directories that contain task result files.

    A directory is considered a model directory if it contains .jsonl files
    matching task patterns (directly or in subdirectories).
    """
    model_dirs = []
    for item in sorted(results_root.iterdir()):
        if not item.is_dir():
            continue
        # Check if this directory has any task JSONL files
        has_tasks = any(item.rglob("task_*.jsonl"))
        if has_tasks:
            model_dirs.append(item)
    return model_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmark pipeline for VLM spatial reasoning tasks"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-dir",
        "-r",
        type=Path,
        help="Path to a single model's results directory",
    )
    group.add_argument(
        "--results-root",
        "-R",
        type=Path,
        help="Path to root directory containing multiple model directories (benchmarks all)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=None,
        help="Path to dataset JSONL (default: read from run_config or use datasets/spatial_data.jsonl)",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        action="append",
        help="Specific task(s) to evaluate (can specify multiple). If not provided, auto-detect.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files (also skips analysis)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only run evaluation (skip analysis)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis (skip evaluation, requires existing evaluation JSONs)",
    )
    parser.add_argument(
        "--with-factor-plots",
        action="store_true",
        help="Generate factor bar chart PNGs (skipped by default, confusion matrices always generated)",
    )

    args = parser.parse_args()

    # Determine what to run
    do_evaluate = not args.analyze_only
    do_analyze = not args.evaluate_only and not args.no_save

    # Collect model directories to process
    if args.results_root:
        if not args.results_root.exists():
            print(f"Error: Results root not found: {args.results_root}")
            sys.exit(1)
        model_dirs = _discover_model_dirs(args.results_root)
        if not model_dirs:
            print(f"Error: No model directories found in {args.results_root}")
            sys.exit(1)
        print(f"Found {len(model_dirs)} model directories to benchmark:")
        for d in model_dirs:
            print(f"  - {d.name}")
        print()
    else:
        model_dirs = [args.results_dir]

    # Process each model directory
    failed_models = []
    for model_dir in model_dirs:
        print("=" * 70)
        print(f"BENCHMARKING: {model_dir.name}")
        print("=" * 70)

        try:
            pipeline = BenchmarkPipeline(model_dir, args.dataset)
            pipeline.run(
                tasks=args.task,
                save=not args.no_save,
                do_evaluate=do_evaluate,
                do_analyze=do_analyze,
                include_factor_plots=args.with_factor_plots,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            failed_models.append((model_dir.name, str(e)))
        except ValueError as e:
            print(f"Error: {e}")
            failed_models.append((model_dir.name, str(e)))
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            failed_models.append((model_dir.name, str(e)))

        print()

    # Summary for multi-model runs
    if args.results_root:
        print("=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)
        successful = len(model_dirs) - len(failed_models)
        print(f"Processed: {successful}/{len(model_dirs)} models successfully")
        if failed_models:
            print("\nFailed models:")
            for name, error in failed_models:
                print(f"  - {name}: {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
