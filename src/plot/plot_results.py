"""
Plot comparison results between base Qwen3 model and fine-tuned version.
Tasks: C (ans accuracy), E (answer + ASCII accuracy), G (answer + ASCII accuracy), H (ans accuracy)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Professional color palette - softer academic colors
COLORS = {
    "base": "#6d3e8c",
    "finetune": "#f2b988",
    "bg_light": "white",  # Light gray for alternating background
}


def load_evaluation_summary(results_dir: Path) -> dict:
    """Load evaluation summary from results directory."""
    summary_path = results_dir / "outputs" / "evaluation" / "evaluation_summary.json"
    with open(summary_path) as f:
        return json.load(f)


def extract_metrics(summary: dict) -> dict:
    """Extract relevant metrics from evaluation summary."""
    tasks = summary["tasks"]

    metrics = {
        "task_c_accuracy": tasks.get("c:C/grid/1000", {}).get("accuracy", 0),
        "task_e_answer_first_ans_acc": tasks.get("e:E/grid/answer_first/1000", {}).get(
            "accuracy", 0
        ),
        "task_e_answer_first_ascii_acc": tasks.get(
            "e:E/grid/answer_first/1000", {}
        ).get("ascii_correctness_rate", 0),
        "task_e_ascii_first_ans_acc": tasks.get("e:E/grid/ascii_first/1000", {}).get(
            "accuracy", 0
        ),
        "task_e_ascii_first_ascii_acc": tasks.get("e:E/grid/ascii_first/1000", {}).get(
            "ascii_correctness_rate", 0
        ),
        "task_g_answer_first_ans_acc": tasks.get("g:G/grid/answer_first/1000", {}).get(
            "accuracy", 0
        ),
        "task_g_answer_first_ascii_acc": tasks.get(
            "g:G/grid/answer_first/1000", {}
        ).get("ascii_correctness_rate", 0),
        "task_g_ascii_first_ans_acc": tasks.get("g:G/grid/ascii_first/1000", {}).get(
            "accuracy", 0
        ),
        "task_g_ascii_first_ascii_acc": tasks.get("g:G/grid/ascii_first/1000", {}).get(
            "ascii_correctness_rate", 0
        ),
    }

    # Task H may have different paths
    h_path_options = ["h:H/grid/1000", "h:H/grid/ascii_first/1000"]
    for path in h_path_options:
        if path in tasks:
            metrics["task_h_accuracy"] = tasks[path].get("accuracy", 0)
            break

    return metrics


def plot_wide_comparison(base_metrics: dict, ft_metrics: dict, output_dir: Path):
    """Create a single wide plot showing all task results."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "pdf.fonttype": 42,  # TrueType fonts for better quality
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    # Define task groups with spacing
    # Format: (label, base_value, ft_value)
    groups = [
        # Task C
        ("Answer", base_metrics["task_c_accuracy"], ft_metrics["task_c_accuracy"]),
        # Task E - grouped by mode
        (
            "Answer\n(Ans First)",
            base_metrics["task_e_answer_first_ans_acc"],
            ft_metrics["task_e_answer_first_ans_acc"],
        ),
        (
            "ASCII\n(Ans First)",
            base_metrics["task_e_answer_first_ascii_acc"],
            ft_metrics["task_e_answer_first_ascii_acc"],
        ),
        (
            "Answer\n(ASCII First)",
            base_metrics["task_e_ascii_first_ans_acc"],
            ft_metrics["task_e_ascii_first_ans_acc"],
        ),
        (
            "ASCII\n(ASCII First)",
            base_metrics["task_e_ascii_first_ascii_acc"],
            ft_metrics["task_e_ascii_first_ascii_acc"],
        ),
        # Task G - grouped by mode
        (
            "Answer\n(Ans First)",
            base_metrics["task_g_answer_first_ans_acc"],
            ft_metrics["task_g_answer_first_ans_acc"],
        ),
        (
            "ASCII\n(Ans First)",
            base_metrics["task_g_answer_first_ascii_acc"],
            ft_metrics["task_g_answer_first_ascii_acc"],
        ),
        (
            "Answer\n(ASCII First)",
            base_metrics["task_g_ascii_first_ans_acc"],
            ft_metrics["task_g_ascii_first_ans_acc"],
        ),
        (
            "ASCII\n(ASCII First)",
            base_metrics["task_g_ascii_first_ascii_acc"],
            ft_metrics["task_g_ascii_first_ascii_acc"],
        ),
    ]

    # Create x positions with gaps between task groups
    # Task C: 1 bar, Task E: 4 bars, Task G: 4 bars
    group_sizes = [1, 4, 4]
    gap = 0.3  # Small gap between groups

    x_positions = []
    current_x = 0
    group_centers = []
    group_starts = []

    for size in group_sizes:
        group_starts.append(current_x)
        for i in range(size):
            x_positions.append(current_x)
            current_x += 1
        group_centers.append((group_starts[-1] + current_x - 1) / 2)
        current_x += gap

    x = np.array(x_positions)
    width = 0.38

    task_labels = [
        "Desc",
        "Desc->Answer & ASCII in One-turn",
        "Desc->Answer & ASCII in Two-turn",
    ]

    # Draw bars
    base_values = [g[1] for g in groups]
    ft_values = [g[2] for g in groups]
    labels = [g[0] for g in groups]

    bars1 = ax.bar(
        x - width / 2,
        base_values,
        width,
        label="Qwen3-30B-A3B",
        color=COLORS["base"],
        zorder=2,
    )
    bars2 = ax.bar(
        x + width / 2,
        ft_values,
        width,
        label="Qwen3-30B-A3B Fine-tuned with Desc->ASCII",
        color=COLORS["finetune"],
        zorder=2,
    )

    # Styling
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, x[-1] + 0.6 + width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray", zorder=0)
    ax.set_axisbelow(True)

    # Add value labels inside bars (dark grey text)
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 0.05,
            f"{height:.0%}",
            ha="center",
            va="top",
            fontsize=9,
            color="white",
            fontweight="bold",
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height - 0.05,
            f"{height:.0%}",
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    # Add task labels at bottom (below x-axis labels)
    for center, label in zip(group_centers, task_labels):
        ax.text(
            center,
            -0.22,
            label,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    # Horizontal legend at top
    ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # Make room for task labels
    plt.savefig(
        output_dir / "qwen3_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.savefig(
        output_dir / "qwen3_comparison.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        backend="pdf",
        dpi=600,
    )
    plt.close()
    print(f"Saved: qwen3_comparison.png/pdf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot base vs fine-tuned metrics from two evaluated run directories."
    )
    parser.add_argument(
        "--base-dir", required=True, help="Run directory of the base model"
    )
    parser.add_argument(
        "--finetune-dir", required=True, help="Run directory of the fine-tuned model"
    )
    parser.add_argument(
        "--output-dir", default="outputs/plots", help="Where to write the figure"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation summaries...")
    base_summary = load_evaluation_summary(Path(args.base_dir))
    ft_summary = load_evaluation_summary(Path(args.finetune_dir))

    base_metrics = extract_metrics(base_summary)
    ft_metrics = extract_metrics(ft_summary)

    print("\n=== Base Model Metrics ===")
    for k, v in base_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Fine-tuned Model Metrics ===")
    for k, v in ft_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nGenerating plot...")
    plot_wide_comparison(base_metrics, ft_metrics, output_dir)
    print(f"\nPlot saved to: {output_dir}")


if __name__ == "__main__":
    main()
