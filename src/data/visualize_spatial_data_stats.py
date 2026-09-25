#!/usr/bin/env uv run python
"""
Visualization script for spatial_data.jsonl
Displays distribution of components, relations, query types, and terminology usage in clean, easy-to-understand plots.

Key visualizations:
- Component and relation count distributions
- Query type distribution (full, vertical, horizontal)
- Terminology usage (spatial, cardinal, clock, and combinations)
- Direct vs indirect relations
- Unique vs non-unique layouts
- Ambiguity tracking statistics
- Summary statistics
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Unified color palette for consistent aesthetics
COLORS = {
    # Primary palette (ordered for multi-category use)
    "primary": ["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"],
    # Binary outcomes (use blue and orange for consistency)
    "binary": ["#3B82F6", "#F59E0B"],  # Blue and orange
    # Bar chart color
    "bar": "#3B82F6",  # Blue
}


def load_data(filepath):
    """Load and parse the JSONL data."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_visualizations(data):
    """Create comprehensive visualizations of the spatial data."""
    # Set up the plotting style with improved aesthetics
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.4})
    sns.set_palette(COLORS["primary"])

    # Create a large figure with subplots (8 plots in 2 rows)
    fig = plt.figure(figsize=(16, 10), facecolor="white")

    # Extract key metrics
    num_components = [d["num_components"] for d in data]
    num_relations = [d["num_relations"] for d in data]
    is_directly_stated = [d["is_directly_stated"] for d in data]
    has_unique_layout = [d["has_unique_layout"] for d in data]
    query_types = [d["query_type"] for d in data]
    ambiguous_stages = [d["ambiguous_stages"] for d in data]
    terminology_used = [d["terminology_used"] for d in data]

    # 1. Component Distribution
    plt.subplot(2, 4, 1)
    component_counts = Counter(num_components)
    components = sorted(component_counts.keys())
    counts = [component_counts[c] for c in components]
    plt.bar(
        components,
        counts,
        color=COLORS["bar"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )
    plt.title("Distribution of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Frequency")

    # 2. Relations Distribution
    plt.subplot(2, 4, 2)
    relation_counts = Counter(num_relations)
    relations = sorted(relation_counts.keys())
    relation_frequencies = [relation_counts[h] for h in relations]
    plt.bar(
        relations,
        relation_frequencies,
        color=COLORS["bar"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )
    plt.title("Distribution of Relations")
    plt.xlabel("Number of Relations")
    plt.ylabel("Frequency")

    # 3. Query Type Distribution
    plt.subplot(2, 4, 3)
    query_type_counts = Counter(query_types)
    labels = ["Full", "Vertical", "Horizontal"]
    values = [
        query_type_counts.get("full", 0),
        query_type_counts.get("vertical", 0),
        query_type_counts.get("horizontal", 0),
    ]
    colors_3 = COLORS["primary"][:3]  # Use first 3 from primary palette
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_3,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    plt.title("Query Type Distribution")

    # 4. Terminology Used in Descriptions
    plt.subplot(2, 4, 4)
    terminology_counts = Counter(terminology_used)

    # Sort by count for better visualization
    sorted_terms = sorted(terminology_counts.items(), key=lambda x: -x[1])
    labels = [term for term, _ in sorted_terms]
    values = [count for _, count in sorted_terms]

    # Use as many colors as needed from primary palette
    colors_n = (
        COLORS["primary"][: len(labels)]
        if len(labels) <= len(COLORS["primary"])
        else COLORS["primary"]
    )
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors_n,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    plt.title("Terminology Distribution")

    # 5. Direct vs Indirect Relations
    plt.subplot(2, 4, 5)
    direct_counts = Counter(is_directly_stated)
    labels = ["Indirect", "Direct"]
    values = [direct_counts[False], direct_counts[True]]
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=COLORS["binary"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    plt.title("Direct vs Indirect Relations")

    # 6. Unique vs Non-Unique Layouts
    plt.subplot(2, 4, 6)
    unique_counts = Counter(has_unique_layout)
    labels = ["Non-Unique", "Unique"]
    values = [unique_counts[False], unique_counts[True]]
    wedges, texts, autotexts = plt.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=COLORS["binary"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    plt.title("Unique vs Non-Unique Layouts")

    # 7. Ambiguous Stages Distribution
    plt.subplot(2, 4, 7)
    ambiguous_counts = Counter(ambiguous_stages)
    stages = sorted(ambiguous_counts.keys())
    stage_frequencies = [ambiguous_counts[s] for s in stages]
    plt.bar(
        stages,
        stage_frequencies,
        color=COLORS["bar"],
        alpha=0.85,
        edgecolor="white",
        linewidth=1.2,
    )
    plt.title("Distribution of Ambiguous Stages")
    plt.xlabel("Number of Ambiguous Stages")
    plt.ylabel("Frequency")

    # 8. Summary statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis("off")

    total_ambiguous = sum(ambiguous_stages)
    avg_ambiguous = total_ambiguous / len(ambiguous_stages) if ambiguous_stages else 0

    # Get top 3 terminology types
    top_terminology = sorted(terminology_counts.items(), key=lambda x: -x[1])[:3]
    term_lines = "\n".join(
        [f"  {term}: {100*count/len(data):.1f}%" for term, count in top_terminology]
    )

    stats_text = f"""DATASET SUMMARY

Total: {len(data):,} instances

Components: {min(num_components)}-{max(num_components)}
  Mean: {np.mean(num_components):.1f}

Relations: {min(num_relations)}-{max(num_relations)}
  Mean: {np.mean(num_relations):.1f}

Query Types:
  Full: {100*query_type_counts.get('full', 0)/len(data):.1f}%
  Vert: {100*query_type_counts.get('vertical', 0)/len(data):.1f}%
  Horiz: {100*query_type_counts.get('horizontal', 0)/len(data):.1f}%

Top Terminology:
{term_lines}

Relations:
  Direct: {100*sum(is_directly_stated)/len(is_directly_stated):.1f}%
  Unique: {100*sum(has_unique_layout)/len(has_unique_layout):.1f}%

Ambiguity: {avg_ambiguous:.2f} avg stages
    """
    ax8.text(
        0.1,
        0.95,
        stats_text,
        transform=ax8.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#F8FAFC",
            edgecolor="#E2E8F0",
            linewidth=1.5,
        ),
    )
    ax8.set_title("Summary Statistics")

    plt.tight_layout(pad=3.0)
    return fig


def print_summary_stats(data):
    """Print detailed summary statistics to console."""
    num_components = [d["num_components"] for d in data]
    num_relations = [d["num_relations"] for d in data]
    is_directly_stated = [d["is_directly_stated"] for d in data]
    has_unique_layout = [d["has_unique_layout"] for d in data]
    query_types = [d["query_type"] for d in data]
    ambiguous_stages = [d["ambiguous_stages"] for d in data]
    terminology_used = [d["terminology_used"] for d in data]

    print("=" * 60)
    print("SPATIAL DATA ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(data):,}")
    print()

    print("COMPONENT DISTRIBUTION:")
    component_counts = Counter(num_components)
    for comp in sorted(component_counts.keys()):
        print(
            f"  {comp} components: {component_counts[comp]:,} records ({100*component_counts[comp]/len(data):.1f}%)"
        )
    print()

    print("RELATION DISTRIBUTION:")
    relation_counts = Counter(num_relations)
    for rel in sorted(relation_counts.keys()):
        print(
            f"  {rel} relations: {relation_counts[rel]:,} records ({100*relation_counts[rel]/len(data):.1f}%)"
        )
    print()

    print("QUERY TYPE DISTRIBUTION:")
    query_type_counts = Counter(query_types)
    for qtype in ["full", "vertical", "horizontal"]:
        count = query_type_counts.get(qtype, 0)
        print(f"  {qtype.capitalize()}: {count:,} records ({100*count/len(data):.1f}%)")
    print()

    print("TERMINOLOGY USED IN DESCRIPTIONS:")
    terminology_counts = Counter(terminology_used)
    sorted_terms = sorted(terminology_counts.items(), key=lambda x: -x[1])
    for term, count in sorted_terms:
        print(f"  {term}: {count:,} records ({100*count/len(data):.1f}%)")
    print()

    print("RELATION CHARACTERISTICS:")
    print(
        f"  Direct relations: {sum(is_directly_stated):,} ({100*sum(is_directly_stated)/len(is_directly_stated):.1f}%)"
    )
    print(
        f"  Indirect relations: {len(is_directly_stated)-sum(is_directly_stated):,} ({100*(len(is_directly_stated)-sum(is_directly_stated))/len(is_directly_stated):.1f}%)"
    )
    print(
        f"  Unique layouts: {sum(has_unique_layout):,} ({100*sum(has_unique_layout)/len(has_unique_layout):.1f}%)"
    )
    print()

    print("AMBIGUITY STATISTICS:")
    total_ambiguous = sum(ambiguous_stages)
    avg_ambiguous = total_ambiguous / len(ambiguous_stages) if ambiguous_stages else 0
    print(f"  Total ambiguous stages: {total_ambiguous:,}")
    print(f"  Average per instance: {avg_ambiguous:.2f}")
    ambiguous_counts = Counter(ambiguous_stages)
    for stage in sorted(ambiguous_counts.keys()):
        print(
            f"    {stage} stages: {ambiguous_counts[stage]:,} instances ({100*ambiguous_counts[stage]/len(data):.1f}%)"
        )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Visualize spatial relationship data")
    parser.add_argument(
        "--input",
        "-i",
        default="datasets/spatial_data.jsonl",
        help="Input JSONL file (default: datasets/spatial_data.jsonl)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="datasets/spatial_analysis",
        help="Output visualization file (without extension, default: datasets/spatial_analysis)",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Print detailed statistics to console",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Output image DPI (default: 300)"
    )
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Output format(s) (default: png pdf)",
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    data = load_data(args.input)

    if args.stats:
        print_summary_stats(data)

    print("Creating visualizations...")
    fig = create_visualizations(data)

    # Remove any existing extension from output path
    output_base = Path(args.output)
    if output_base.suffix in [".png", ".pdf", ".svg"]:
        output_base = output_base.with_suffix("")

    # Save in all requested formats
    for fmt in args.format:
        output_path = f"{output_base}.{fmt}"
        print(f"Saving visualization to {output_path}...")
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")

    print("Done!")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
