"""CLI for aggregating benchmark results into dashboard data."""

import argparse
from pathlib import Path

from .aggregator import DashboardAggregator


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark results for dashboard visualization"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing model results (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dashboard_data.json"),
        help="Output JSON file path (default: dashboard_data.json)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print discovery progress",
    )

    args = parser.parse_args()

    aggregator = DashboardAggregator(args.results_dir)

    if args.verbose:
        print(f"Scanning {args.results_dir} for model results...")

    models = aggregator.discover_models()

    if not models:
        print(f"No model results found in {args.results_dir}")
        print(
            "Expected structure: {results_dir}/{model}/outputs/evaluation/evaluation_summary.json"
        )
        return

    if args.verbose:
        print(f"Found {len(models)} models: {', '.join(models)}")

    data = aggregator.aggregate_all()
    aggregator.to_json(data, args.output)

    print(f"Aggregated {len(data.models)} models to {args.output}")
    for model_name in data.models:
        model = data.models[model_name]
        task_count = len(model.get("summary", {}).get("tasks", {}))
        print(f"  - {model_name}: {task_count} tasks")


if __name__ == "__main__":
    main()
