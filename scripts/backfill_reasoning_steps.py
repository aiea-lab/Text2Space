"""Backfill reasoning_steps field into an existing spatial_data.jsonl dataset.

Uses ASCIIEvaluator's query parser and phrase lookup to reconstruct each
instance's SpatialMap, then computes chain-of-thought reasoning steps.

Usage:
    uv run python scripts/backfill_reasoning_steps.py
    uv run python scripts/backfill_reasoning_steps.py --input datasets/custom.jsonl --output datasets/custom_with_reasoning.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.spatial_constants import (
    CARDINAL_SYNONYMS,
    CLOCK_SYNONYMS,
    DIRECTION_SYNONYMS,
    INVERSE_DIRECTIONS,
)
from data.spatial_map import SpatialMap
from evaluation.ascii_evaluator import ASCIIEvaluator


def build_phrase_to_direction() -> Dict[str, str]:
    """Build reverse lookup: phrase -> base direction from all synonym dicts."""
    lookup = {}
    for synonyms in (DIRECTION_SYNONYMS, CARDINAL_SYNONYMS, CLOCK_SYNONYMS):
        for direction, phrases in synonyms.items():
            for phrase in phrases:
                lookup[phrase.lower()] = direction
    return lookup


def parse_stated_relations(
    description: str, phrase_lookup: Dict[str, str]
) -> List[Tuple[str, str, str]]:
    """Parse description into stated (obj1, obj2, direction) tuples.

    Unlike ASCIIEvaluator.parse_description which splits on 'and' (breaking
    compound directions like 'above and to the right of'), this treats the
    entire phrase between 'is' and the final object as one lookup key.
    """
    relations = []
    sentences = [s.strip().rstrip(".") for s in description.split(".") if s.strip()]

    for sentence in sentences:
        is_idx = sentence.find(" is ")
        if is_idx == -1:
            continue
        obj1 = sentence[:is_idx].strip()
        rest = sentence[is_idx + 4 :]  # skip " is "

        # obj2 is the last word; phrase is everything before it
        parts = rest.rsplit(None, 1)
        if len(parts) < 2:
            continue
        phrase, obj2 = parts[0].lower(), parts[1].strip()

        direction = phrase_lookup.get(phrase)
        if direction:
            relations.append((obj1, obj2, direction))

    return relations


def backfill(input_path: str, output_path: str) -> None:
    """Add reasoning_steps to each instance and write to a new file."""
    in_path = Path(input_path)
    if not in_path.exists():
        print(f"Error: {in_path} not found")
        sys.exit(1)

    evaluator = ASCIIEvaluator()
    phrase_lookup = build_phrase_to_direction()

    instances = []
    with open(in_path) as f:
        for line in f:
            instances.append(json.loads(line))

    print(f"Processing {len(instances)} instances...")

    failures = 0
    for i, inst in enumerate(instances):
        # Parse stated relations from description
        relations = parse_stated_relations(inst["description"], phrase_lookup)
        if not relations:
            inst["reasoning_steps"] = ""
            failures += 1
            continue

        # Reconstruct SpatialMap graph directly from stated relations.
        # We bypass add_relation() because it enforces geometric consistency
        # based on insertion order, which may differ from the original generation.
        spatial_map = SpatialMap()
        for obj1, obj2, direction in relations:
            inverse_dir = INVERSE_DIRECTIONS[direction]
            # relations[ref] = [(target, dir)] means "target is dir from ref"
            spatial_map.relations[obj2].append((obj1, direction))
            spatial_map.relations[obj1].append((obj2, inverse_dir))
        spatial_map._objects_cache = None  # invalidate so .objects recomputes

        # Parse query to get comp1 and comp2
        comp1, comp2, _ = evaluator.parse_query(inst["query_relation"])
        if comp1 is None:
            inst["reasoning_steps"] = ""
            failures += 1
            continue

        # Compute reasoning steps (parse_query returns lowercase, SpatialMap uses uppercase)
        inst["reasoning_steps"] = spatial_map.get_reasoning_steps(
            comp1.upper(), comp2.upper()
        )

        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(instances)} done")

    # Write to new file with reasoning_steps inserted after label
    out_path = Path(output_path)
    with open(out_path, "w") as f:
        for inst in instances:
            ordered = {}
            for key, value in inst.items():
                if key == "reasoning_steps":
                    continue
                ordered[key] = value
                if key == "label":
                    ordered["reasoning_steps"] = inst.get("reasoning_steps", "")
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")

    print(f"Done. {len(instances)} instances written to {out_path}")
    if failures:
        print(f"  {failures} instances had empty reasoning_steps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill reasoning_steps into dataset"
    )
    parser.add_argument(
        "--input",
        default="datasets/spatial_data.jsonl",
        help="Path to input JSONL dataset",
    )
    parser.add_argument(
        "--output",
        default="datasets/spatial_data_with_reasoning.jsonl",
        help="Path to output JSONL dataset",
    )
    args = parser.parse_args()
    backfill(args.input, args.output)
