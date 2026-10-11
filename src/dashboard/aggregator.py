"""Aggregate benchmark results from multiple models for dashboard visualization."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class DashboardData:
    """Aggregated dashboard data structure."""

    generated_at: str
    models: dict = field(default_factory=dict)
    all_tasks: list = field(default_factory=list)


class DashboardAggregator:
    """Discovers and aggregates model results for the dashboard."""

    def __init__(self, results_dir: Path = Path("results")):
        self.results_dir = Path(results_dir)

    def discover_models(self) -> list[str]:
        """Find all directories with outputs/evaluation/evaluation_summary.json."""
        models = []
        if not self.results_dir.exists():
            return models

        for item in self.results_dir.iterdir():
            if item.is_dir():
                summary_path = (
                    item / "outputs" / "evaluation" / "evaluation_summary.json"
                )
                if summary_path.exists():
                    models.append(item.name)

        return sorted(models)

    def _load_json(self, path: Path) -> dict | None:
        """Load a JSON file, returning None if it doesn't exist."""
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _format_display_name(self, model_name: str) -> str:
        """Convert directory name to human-readable display name."""
        # Remove common prefixes/suffixes and format nicely
        name = model_name.replace("_", " ").replace("-", " ")
        # Capitalize each word
        return " ".join(word.capitalize() for word in name.split())

    def _find_confusion_matrix_paths(
        self, model_dir: Path
    ) -> dict[str, dict[str, str]]:
        """Find confusion matrix PNG paths for each task, grouped by query type.

        Returns:
            Dict mapping task_id -> {query_type -> path}
            where query_type is one of: combined, full, horizontal, vertical
        """
        matrices = {}
        analysis_dir = model_dir / "outputs" / "analysis"

        # Query type suffixes to look for (in display order)
        query_types = ["combined", "full", "horizontal", "vertical"]

        if analysis_dir.exists():
            for png_file in analysis_dir.glob("task_*_confusion_matrix*.png"):
                stem = (
                    png_file.stem
                )  # e.g., "task_c_confusion_matrix" or "task_c_confusion_matrix_full"
                rel_path = str(png_file.relative_to(self.results_dir.parent))

                # Check for query-type-specific files first
                query_type = None
                for qt in ["full", "horizontal", "vertical"]:
                    if stem.endswith(f"_confusion_matrix_{qt}"):
                        query_type = qt
                        # Extract task_id: remove "task_" prefix and "_confusion_matrix_{qt}" suffix
                        suffix_len = len(f"_confusion_matrix_{qt}")
                        task_id = stem[5:-suffix_len]
                        break

                # If no query type suffix, it's the combined matrix
                if query_type is None and "_confusion_matrix" in stem:
                    # Make sure it's not a partial match (e.g., ends exactly with _confusion_matrix)
                    if stem.endswith("_confusion_matrix"):
                        query_type = "combined"
                        task_id = stem[
                            5:-17
                        ]  # Remove "task_" (5) and "_confusion_matrix" (17)

                if query_type and task_id:
                    if task_id not in matrices:
                        matrices[task_id] = {}
                    matrices[task_id][query_type] = rel_path

        return matrices

    def _find_factor_plot_paths(self, model_dir: Path) -> dict[str, list[str]]:
        """Find factor analysis PNG paths grouped by task."""
        plots = {}
        analysis_dir = model_dir / "outputs" / "analysis"

        if analysis_dir.exists():
            for png_file in analysis_dir.glob("*.png"):
                # Skip confusion matrices
                if "confusion_matrix" in png_file.name:
                    continue

                # Extract task letter from filename like "a_num_components.png"
                parts = png_file.stem.split("_")
                if len(parts) >= 2:
                    task_letter = parts[0]
                    if task_letter not in plots:
                        plots[task_letter] = []
                    plots[task_letter].append(
                        str(png_file.relative_to(self.results_dir.parent))
                    )

        return plots

    def _load_task_e_bucket_analysis(self, model_dir: Path) -> dict:
        """Load bucket analysis and error mapping for Task E variants."""
        result = {}
        eval_dir = model_dir / "outputs" / "evaluation"

        if not eval_dir.exists():
            return result

        # Look for task_e_*_evaluation.json files
        for eval_file in eval_dir.glob("task_e_*_evaluation.json"):
            data = self._load_json(eval_file)
            if not data:
                continue

            # Extract variant from filename: task_e_E_grid_answer_first_1000_evaluation.json
            filename = (
                eval_file.stem
            )  # e.g., task_e_E_grid_answer_first_1000_evaluation
            if "answer_first" in filename:
                variant = "e_answer_first"
            elif "ascii_first" in filename:
                variant = "e_ascii_first"
            else:
                continue

            # Extract by_ascii_accuracy breakdown (bucket analysis)
            by_ascii = data.get("breakdown", {}).get("by_ascii_accuracy", {})
            if by_ascii:
                result[variant] = by_ascii

            # Extract error_mapping (ASCII correctness × Answer correctness)
            error_mapping = data.get("breakdown", {}).get("error_mapping", {})
            if error_mapping:
                # Store error mapping under a separate key with variant suffix
                result[f"{variant}_error_mapping"] = error_mapping

        return result

    def _load_classification_metrics(self, model_dir: Path) -> dict:
        """Load classification metrics (F1 scores) for all tasks."""
        result = {}
        eval_dir = model_dir / "outputs" / "evaluation"

        if not eval_dir.exists():
            return result

        # Look for all task evaluation files
        for eval_file in eval_dir.glob("task_*_evaluation.json"):
            data = self._load_json(eval_file)
            if not data or not isinstance(data, dict):
                continue

            breakdown = data.get("breakdown", {})
            classification_metrics = breakdown.get("classification_metrics", {})
            classification_metrics_by_qt = breakdown.get(
                "classification_metrics_by_query_type", {}
            )

            if not classification_metrics:
                continue

            # Extract task identifier from filename
            # e.g., task_d_evaluation.json -> d
            # e.g., task_e_E_grid_answer_first_1000_evaluation.json -> e_answer_first
            filename = eval_file.stem
            if filename.startswith("task_"):
                # Remove "task_" prefix and "_evaluation" suffix
                task_id = filename[5:]  # Remove "task_"
                if task_id.endswith("_evaluation"):
                    task_id = task_id[:-11]  # Remove "_evaluation"

                # Normalize task_id for E and G variants
                if "answer_first" in task_id:
                    if task_id.startswith("e"):
                        task_id = "e_answer_first"
                    elif task_id.startswith("g"):
                        task_id = "g_answer_first"
                elif "ascii_first" in task_id:
                    if task_id.startswith("e"):
                        task_id = "e_ascii_first"
                    elif task_id.startswith("g"):
                        task_id = "g_ascii_first"
                else:
                    # Simple task like c, d, f, h
                    task_id = task_id.split("_")[0]  # Take first part only

                result[task_id] = {
                    "overall": {
                        "macro_f1": classification_metrics.get("macro_f1", 0),
                        "micro_f1": classification_metrics.get("micro_f1", 0),
                        "weighted_f1": classification_metrics.get("weighted_f1", 0),
                        "macro_precision": classification_metrics.get(
                            "macro_precision", 0
                        ),
                        "macro_recall": classification_metrics.get("macro_recall", 0),
                    },
                    "per_class": classification_metrics.get("per_class", {}),
                    "by_query_type": {
                        qt: {
                            "macro_f1": metrics.get("macro_f1", 0),
                            "micro_f1": metrics.get("micro_f1", 0),
                            "per_class": metrics.get("per_class", {}),
                        }
                        for qt, metrics in classification_metrics_by_qt.items()
                    },
                }

        return result

    def aggregate_model(self, model_name: str) -> dict:
        """Aggregate all data for a single model."""
        model_dir = self.results_dir / model_name

        # Load evaluation summary
        summary_path = model_dir / "outputs" / "evaluation" / "evaluation_summary.json"
        summary = self._load_json(summary_path) or {}

        # Load factor analysis
        factor_path = model_dir / "outputs" / "analysis" / "factor_analysis.json"
        factor_analysis = self._load_json(factor_path) or {}

        # Find confusion matrix image paths
        confusion_matrices = self._find_confusion_matrix_paths(model_dir)

        # Find factor plot paths
        factor_plots = self._find_factor_plot_paths(model_dir)

        # Load Task E bucket analysis
        task_e_bucket_analysis = self._load_task_e_bucket_analysis(model_dir)

        # Load classification metrics (F1 scores)
        classification_metrics = self._load_classification_metrics(model_dir)

        return {
            "name": model_name,
            "display_name": self._format_display_name(model_name),
            "results_dir": str(model_dir.relative_to(self.results_dir.parent)),
            "summary": summary,
            "factor_analysis": factor_analysis,
            "confusion_matrices": confusion_matrices,
            "factor_plots": factor_plots,
            "task_e_bucket_analysis": task_e_bucket_analysis,
            "classification_metrics": classification_metrics,
        }

    def aggregate_all(self) -> DashboardData:
        """Aggregate all discovered models into DashboardData."""
        models = {}
        all_tasks = set()

        for model_name in self.discover_models():
            model_data = self.aggregate_model(model_name)
            models[model_name] = model_data

            # Collect all task names from this model's summary
            if "tasks" in model_data.get("summary", {}):
                for task_key in model_data["summary"]["tasks"]:
                    # Extract task letter (e.g., "a" from "a:A/grid/1000")
                    task = task_key.split(":")[0] if ":" in task_key else task_key
                    all_tasks.add(task)

        # Sort tasks alphabetically
        sorted_tasks = sorted(all_tasks)

        return DashboardData(
            generated_at=datetime.now().isoformat(),
            models=models,
            all_tasks=sorted_tasks,
        )

    def to_json(self, data: DashboardData, output_path: Path) -> None:
        """Serialize DashboardData to JSON file."""
        output_path = Path(output_path)
        with open(output_path, "w") as f:
            json.dump(asdict(data), f, indent=2)
