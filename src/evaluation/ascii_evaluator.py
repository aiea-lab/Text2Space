#!/usr/bin/env python3
"""
ASCII Grid Evaluator

Compares spatial relationships in ASCII grids. Supports simple, grid, and panel formats.

Evaluation modes:
1. evaluate_grids()       - Compare all pairwise relations between reference and model grids
2. evaluate_description() - Validate stated relations (verify_ascii) or completeness (verify_description)
3. evaluate_query()       - Validate a single spatial query (full, vertical, or horizontal)
4. evaluate_label()       - Check if model output matches ground truth with alias support
"""

import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.spatial_constants import (
    BASE_DIRECTIONS,
    CARDINAL_SYNONYMS,
    CLOCK_SYNONYMS,
    DIRECTION_SYNONYMS,
    LABEL_ALIASES,
    SYNONYM_TO_BASE,
)
from data.spatial_utils import (
    get_horizontal_relation,
    get_relation_from_positions,
    get_vertical_relation,
    normalize_positions,
)


class ASCIIGridParser:
    """Extracts object positions from ASCII grids in any format (simple/grid/panel)."""

    # Characters used in grid borders (ignored during parsing)
    GRID_CHARS = {"+", "-", "|", "─", "│", "┌", "┐", "└", "┘"}

    @staticmethod
    def parse(ascii_grid: str) -> Dict[str, Tuple[int, int]]:
        """Extract object positions from ASCII grid.

        Args:
            ascii_grid: ASCII representation of spatial layout

        Returns:
            Dictionary mapping object names (lowercase) to (x, y) positions

        Raises:
            ValueError: If duplicate objects are found
        """
        positions = {}
        duplicates = []

        # Remove leading/trailing empty lines while preserving indentation
        grid_lines = ascii_grid.split("\n")
        while grid_lines and not grid_lines[0].strip():
            grid_lines.pop(0)
        while grid_lines and not grid_lines[-1].strip():
            grid_lines.pop()
        lines = grid_lines

        for y, line in enumerate(lines):
            i = 0

            while i < len(line):
                char = line[i]

                # Skip whitespace and grid characters
                if char.isspace() or char in ASCIIGridParser.GRID_CHARS:
                    i += 1
                    continue

                # Extract alphanumeric object name
                if char.isalnum():
                    obj_name = ""
                    start_x = i
                    while i < len(line) and line[i].isalnum():
                        obj_name += line[i]
                        i += 1

                    obj_name_lower = obj_name.lower()

                    if obj_name_lower in positions:
                        duplicates.append(obj_name)
                    else:
                        positions[obj_name_lower] = (start_x, y)
                else:
                    i += 1

        if duplicates:
            raise ValueError(
                f"Duplicate objects found in grid: {', '.join(duplicates)}"
            )

        return positions


class SpatialRelationExtractor:
    """Computes spatial relationships between object positions.

    Uses graph-based structure (adjacency list) consistent with SpatialMap.

    Note: Coordinate System Convention
    - ASCII grids use (x, y) where y increases DOWNWARD (row index)
    - SpatialMap uses (x, y) where y increases DOWNWARD internally
    - Both systems are compatible: larger y = lower position
    """

    @staticmethod
    def is_valid_direction(direction: str) -> bool:
        """Check if a direction is valid.

        Matches SpatialMap's direction validation logic.

        Args:
            direction: Direction string to validate

        Returns:
            True if direction is one of the 8 base directions
        """
        return direction in BASE_DIRECTIONS

    @staticmethod
    def extract_all_relations(
        positions: Dict[str, Tuple[int, int]],
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Extract all pairwise spatial relationships as a graph structure.

        Returns:
            Graph structure matching SpatialMap format:
            {obj: [(neighbor, direction), ...]}

        Example:
            {'a': [('b', 'above'), ('c', 'left')],
             'b': [('a', 'below')],
             'c': [('a', 'right')]}
        """
        relations = defaultdict(list)
        objects = list(positions.keys())

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i + 1 :]:
                # Calculate bidirectional relations using shared utility
                relation_1_to_2 = get_relation_from_positions(
                    positions[obj1], positions[obj2]
                )
                relation_2_to_1 = get_relation_from_positions(
                    positions[obj2], positions[obj1]
                )

                # Build graph structure: obj -> [(neighbor, direction), ...]
                relations[obj1].append((obj2, relation_1_to_2))
                relations[obj2].append((obj1, relation_2_to_1))

        return dict(relations)


class ASCIIEvaluator:
    """Evaluates model-generated ASCII grids by comparing spatial relationships."""

    def __init__(self):
        self.parser = ASCIIGridParser()
        self.relation_extractor = SpatialRelationExtractor()
        # Build reverse lookup for parsing descriptions
        # Use DIRECTION_SYNONYMS, CARDINAL_SYNONYMS, and CLOCK_SYNONYMS (used in generation) for parsing
        self._phrase_to_direction = {}

        # Add all spatial synonyms from DIRECTION_SYNONYMS
        for direction, phrases in DIRECTION_SYNONYMS.items():
            for phrase in phrases:
                self._phrase_to_direction[phrase.lower()] = direction

        # Add all cardinal synonyms from CARDINAL_SYNONYMS (includes compound phrases)
        for direction, phrases in CARDINAL_SYNONYMS.items():
            for phrase in phrases:
                self._phrase_to_direction[phrase.lower()] = direction

        # Add all clock synonyms from CLOCK_SYNONYMS
        for direction, phrases in CLOCK_SYNONYMS.items():
            for phrase in phrases:
                self._phrase_to_direction[phrase.lower()] = direction

        # Also add LABEL_ALIASES for backward compatibility and model output matching
        for direction, phrases in LABEL_ALIASES.items():
            for phrase in phrases:
                self._phrase_to_direction[phrase.lower()] = direction

    def evaluate_label(self, model_output: str, ground_truth_label: str) -> Dict:
        """Check if model output matches ground truth label using LABEL_ALIASES.

        Handles common variations in model output:
        - Case insensitive matching
        - Trailing punctuation (. ! ? ; : ,)
        - Labels embedded in sentences

        Args:
            model_output: Raw model output (e.g., "Below.", "The answer is below")
            ground_truth_label: Ground truth spatial relation (e.g., "above", "below")

        Returns:
            Dict with keys:
            - correct: True if predicted direction matches ground truth
            - predicted_label: Normalized base direction from model output
            - ground_truth_label: The ground truth label provided
        """
        # Normalize model output: lowercase, strip whitespace, remove trailing punctuation
        normalized_output = model_output.strip().lower().rstrip(".!?;:,")

        # Normalize ground truth label
        normalized_ground_truth = ground_truth_label.lower()

        # Determine what direction the model predicted by finding the longest matching alias
        # Sort aliases by length (descending) to prefer "upper-left" over "upper"
        all_aliases_sorted = sorted(SYNONYM_TO_BASE.keys(), key=len, reverse=True)
        predicted_base = None

        # First check if the whole normalized output is a known alias
        if normalized_output in SYNONYM_TO_BASE:
            predicted_base = SYNONYM_TO_BASE[normalized_output]
        else:
            # Search for longest matching alias in output using word boundaries
            for alias in all_aliases_sorted:
                pattern = r"\b" + re.escape(alias) + r"\b"
                if re.search(pattern, normalized_output):
                    predicted_base = SYNONYM_TO_BASE[alias]
                    break

        # Correct if predicted base direction matches ground truth
        is_correct = (
            (predicted_base == normalized_ground_truth) if predicted_base else False
        )

        return {
            "correct": is_correct,
            "predicted_label": predicted_base if predicted_base else normalized_output,
            "ground_truth_label": normalized_ground_truth,
        }

    @staticmethod
    def _get_relation_from_graph(
        relations_graph: Dict[str, List[Tuple[str, str]]], obj1: str, obj2: str
    ) -> Optional[str]:
        """Look up a specific relation from the graph structure.

        Args:
            relations_graph: Graph {obj: [(neighbor, direction), ...]}
            obj1: First object
            obj2: Second object (neighbor of obj1)

        Returns:
            Direction string if relation exists, None otherwise
        """
        if obj1 not in relations_graph:
            return None

        for neighbor, direction in relations_graph[obj1]:
            if neighbor == obj2:
                return direction

        return None

    @staticmethod
    def _compare_relation_graphs(
        ref_relations: Dict[str, List[Tuple[str, str]]],
        model_relations: Dict[str, List[Tuple[str, str]]],
    ) -> List[Dict]:
        """Compare two relation graphs and return list of incorrect relations.

        Args:
            ref_relations: Reference graph {obj: [(neighbor, direction), ...]}
            model_relations: Model graph {obj: [(neighbor, direction), ...]}

        Returns:
            List of incorrect relation dictionaries with keys:
            'pair', 'expected', 'actual'
        """
        incorrect_relations = []

        # Check all relations in reference graph
        for obj1, neighbors in ref_relations.items():
            for obj2, ref_direction in neighbors:
                # Find corresponding relation in model graph
                model_direction = ASCIIEvaluator._get_relation_from_graph(
                    model_relations, obj1, obj2
                )

                # Compare relations
                if model_direction != ref_direction:
                    incorrect_relations.append(
                        {
                            "pair": (obj1, obj2),
                            "expected": ref_direction,
                            "actual": model_direction if model_direction else "unknown",
                        }
                    )

        return incorrect_relations

    def evaluate_grids(self, reference_grid: str, model_grid: str) -> Dict:
        """Compare spatial relationships between reference and model grids.

        Compares relationships rather than exact positions, enabling
        dimension-agnostic evaluation.

        Returns:
            Dict with keys: correct, accuracy, total_relations, correct_count,
            incorrect_count, incorrect_relations. On error: correct=False, error,
            and possibly missing_objects/extra_objects.
        """
        try:
            ref_positions = self.parser.parse(reference_grid)
        except ValueError as e:
            return {"correct": False, "error": f"Reference grid error: {str(e)}"}

        try:
            model_positions = self.parser.parse(model_grid)
        except ValueError as e:
            return {"correct": False, "error": f"Model grid error: {str(e)}"}

        ref_objects = set(ref_positions.keys())
        model_objects = set(model_positions.keys())

        missing_objects = ref_objects - model_objects
        extra_objects = model_objects - ref_objects

        if missing_objects or extra_objects:
            return {
                "correct": False,
                "error": "Object mismatch",
                "missing_objects": sorted(missing_objects),
                "extra_objects": sorted(extra_objects),
            }

        ref_normalized = normalize_positions(ref_positions)
        model_normalized = normalize_positions(model_positions)

        ref_relations = self.relation_extractor.extract_all_relations(ref_normalized)
        model_relations = self.relation_extractor.extract_all_relations(
            model_normalized
        )

        # Use graph comparison method
        incorrect_relations = self._compare_relation_graphs(
            ref_relations, model_relations
        )

        # Calculate total relations from graph structure
        total_relations = sum(len(neighbors) for neighbors in ref_relations.values())
        correct_count = total_relations - len(incorrect_relations)
        accuracy = correct_count / total_relations if total_relations > 0 else 0.0

        return {
            "correct": len(incorrect_relations) == 0,
            "accuracy": accuracy,
            "total_relations": total_relations,
            "correct_count": correct_count,
            "incorrect_count": len(incorrect_relations),
            "incorrect_relations": incorrect_relations,
        }

    def parse_description(
        self, description: str
    ) -> Tuple[List[Tuple[str, str, str]], int]:
        """Extract spatial relations from natural language description, including compound relations with 'and'."""
        relations = []

        # Split by periods and commas to get sentences
        raw_sentences = re.split(r"[.,]", description)
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        for sentence in sentences:
            # Extract the leading object
            obj1_match = re.match(r"^(\w+)\s+is\s+", sentence, re.IGNORECASE)
            if not obj1_match:
                continue  # Skip sentences that don't match

            obj1 = obj1_match.group(1).lower()
            rest = sentence[obj1_match.end() :].strip()

            # Use regex to find all "<direction> ... <obj2>" patterns joined by 'and'
            # Matches: "right of B", "above G", etc.
            # Handles unlimited 'and's
            parts = re.split(r"\s+and\s+", rest)
            for part in parts:
                direction, obj2 = self._parse_relation_part(part)
                if direction and obj2:
                    relations.append((obj1, obj2, direction))

        return relations, len(relations)

    def _parse_relation_part(self, part: str) -> Tuple[str, str]:
        """Try to extract direction and object from a relation part.

        Matches patterns like:
            - "<direction> of <obj2>"
            - "<direction> <obj2>"
        """
        part = part.lower().strip()

        # Pattern 1: "<direction> of <obj2>" or "<direction> from <obj2>"
        match = re.match(r"(.+?)\s+(?:of|from|as)\s+(\w+)$", part)
        if match:
            phrase, obj2 = match.group(1).strip(), match.group(2).lower()
            direction = (
                self._phrase_to_direction.get(phrase)
                or self._phrase_to_direction.get(phrase + " of")
                or self._phrase_to_direction.get(phrase + " from")
            )
            if direction:
                return direction, obj2
            elif phrase in [
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
            ]:
                return phrase, obj2

        # Pattern 2: "<direction> <obj2>"
        match_simple = re.match(r"(.+?)\s+(\w+)$", part)
        if match_simple:
            phrase, obj2 = match_simple.group(1).strip(), match_simple.group(2).lower()
            direction = (
                self._phrase_to_direction.get(phrase)
                or self._phrase_to_direction.get(phrase + " of")
                or self._phrase_to_direction.get(phrase + " from")
            )
            if direction:
                return direction, obj2
            elif phrase in [
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
            ]:
                return phrase, obj2

        # If nothing matches
        return None, None

    @staticmethod
    def parse_query(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract object names and query type from query.

        Expected formats:
        - Full spatial: "Where is <obj1> relative to <obj2>?"
        - Vertical: "Where is <obj1> vertically relative to <obj2>?"
        - Horizontal: "Where is <obj1> horizontally relative to <obj2>?"

        Returns: (obj1, obj2, query_type) or (None, None, None) if parsing fails
        query_type is one of: "full", "vertical", "horizontal"
        """
        # Try vertical pattern first
        pattern_vertical = r"Where is (\w+) vertically relative to (\w+)\?"
        match = re.search(pattern_vertical, query, re.IGNORECASE)
        if match:
            obj1 = match.group(1).lower()
            obj2 = match.group(2).lower()
            return obj1, obj2, "vertical"

        # Try horizontal pattern
        pattern_horizontal = r"Where is (\w+) horizontally relative to (\w+)\?"
        match = re.search(pattern_horizontal, query, re.IGNORECASE)
        if match:
            obj1 = match.group(1).lower()
            obj2 = match.group(2).lower()
            return obj1, obj2, "horizontal"

        # Try full spatial pattern (no modifier)
        pattern_full = r"Where is (\w+) relative to (\w+)\?"
        match = re.search(pattern_full, query, re.IGNORECASE)
        if match:
            obj1 = match.group(1).lower()
            obj2 = match.group(2).lower()
            return obj1, obj2, "full"

        return None, None, None

    def evaluate_description(
        self, model_grid: str, description: str, mode: str = "verify_ascii"
    ) -> Dict:
        """Validate spatial relationships between description and ASCII grid.

        Supports two evaluation modes:
        - 'verify_ascii': Verify ASCII matches stated relations in description.
          Use when description is ground truth. (Default, original behavior)
        - 'verify_description': Verify description is correct against ASCII.
          Use when ASCII is ground truth. Only checks if stated relations are correct.

        Args:
            model_grid: ASCII grid representation
            description: Natural language description of spatial relationships
            mode: Evaluation mode ('verify_ascii' or 'verify_description')

        Returns:
            For 'verify_ascii' mode:
                Dict with keys: correct, accuracy, total_relations, correct_count,
                incorrect_count, incorrect_relations, correct_relations.
            For 'verify_description' mode:
                Dict with keys: correct, accuracy, total_relations, correct_count,
                incorrect_count, correct_relations, incorrect_relations.
        """
        # Validate mode parameter
        if mode not in ("verify_ascii", "verify_description"):
            return {
                "correct": False,
                "accuracy": 0.0,
                "error": f"Invalid mode '{mode}'. Must be 'verify_ascii' or 'verify_description'",
            }

        stated_relations, total_relations = self.parse_description(description)

        # For verify_ascii mode, empty description is an error
        # For verify_description mode, empty description means all relations are missing
        if total_relations == 0 and mode == "verify_ascii":
            return {
                "correct": False,
                "accuracy": 0.0,
                "error": "No sentences found in description",
                "description": description,
            }

        try:
            model_positions = self.parser.parse(model_grid)
        except ValueError as e:
            return {
                "correct": False,
                "accuracy": 0.0,
                "error": f"Model grid parsing error: {str(e)}",
                "total_relations": total_relations,
            }

        model_normalized = normalize_positions(model_positions)
        model_relations = self.relation_extractor.extract_all_relations(
            model_normalized
        )

        # Extract all objects from successfully parsed relations
        described_objects = set()
        for obj1, obj2, _ in stated_relations:
            described_objects.add(obj1)
            described_objects.add(obj2)

        model_objects = set(model_normalized.keys())
        extra_objects = model_objects - described_objects

        # Distinguish between parsing failures vs truly extra objects
        unparseable_objects = set()
        truly_extra_objects = set()
        description_lower = description.lower()

        for obj in extra_objects:
            # Check if object appears as a standalone word in description
            if re.search(r"\b" + re.escape(obj) + r"\b", description_lower):
                unparseable_objects.add(obj)
            else:
                truly_extra_objects.add(obj)

        # Evaluate all successfully parsed relations
        validation_failures = []
        correct_relations = []
        missing_objects = []

        for obj1, obj2, expected_direction in stated_relations:
            if obj1 not in model_normalized:
                missing_objects.append(obj1)
                validation_failures.append(
                    {
                        "relation": f"{obj1} is {expected_direction} of {obj2}",
                        "error": f'Object "{obj1}" not found in model grid',
                    }
                )
                continue

            if obj2 not in model_normalized:
                missing_objects.append(obj2)
                validation_failures.append(
                    {
                        "relation": f"{obj1} is {expected_direction} of {obj2}",
                        "error": f'Object "{obj2}" not found in model grid',
                    }
                )
                continue

            # Validate relation from graph structure
            actual_direction = self._get_relation_from_graph(
                model_relations, obj1, obj2
            )

            if actual_direction != expected_direction:
                validation_failures.append(
                    {
                        "relation": f"{obj1} is {expected_direction} of {obj2}",
                        "expected": expected_direction,
                        "actual": actual_direction if actual_direction else "unknown",
                        "objects": (obj1, obj2),
                    }
                )
            else:
                # Relation is correct - store direction in base format
                correct_relations.append(expected_direction)

        # Calculate counts
        num_parsed = len(stated_relations)
        num_unparsed = total_relations - num_parsed
        num_validation_failures = len(validation_failures)

        if mode == "verify_description":
            # Calculate counts based on stated relations only
            total_stated = len(stated_relations)
            correct_count = total_stated - num_validation_failures

            # Accuracy = correct / total stated relations
            # Only checks if stated relations are correct, no penalty for missing relations
            accuracy = correct_count / total_stated if total_stated > 0 else 0.0

            # Correct only if no incorrect relations AND no missing objects
            is_correct = num_validation_failures == 0 and not missing_objects

            result = {
                "correct": is_correct,
                "accuracy": accuracy,
                "total_relations": total_stated,
                "correct_count": correct_count,
                "incorrect_count": num_validation_failures,
                "correct_relations": correct_relations,
                "incorrect_relations": validation_failures,
            }

            # Add objects mentioned in description but not found in grid
            if missing_objects:
                result["objects_not_in_grid"] = sorted(set(missing_objects))

        else:  # mode == 'verify_ascii'
            # Correct = successfully parsed AND validated correctly
            correct_count = num_parsed - num_validation_failures

            # Incorrect = validation failures + parsing failures
            incorrect_count = num_validation_failures + num_unparsed

            # Accuracy based on total sentences
            accuracy = correct_count / total_relations if total_relations > 0 else 0.0

            # Fully correct only if ALL sentences parsed and validated
            is_correct = (
                num_validation_failures == 0
                and num_unparsed == 0
                and not unparseable_objects
                and not truly_extra_objects
                and not missing_objects
            )

            result = {
                "correct": is_correct,
                "accuracy": accuracy,
                "total_relations": total_relations,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "incorrect_relations": validation_failures,
                "correct_relations": correct_relations,
            }

            # Add parsing failure info
            if num_unparsed > 0:
                result["unparsed_count"] = num_unparsed
                if unparseable_objects:
                    result["unparseable_objects"] = sorted(unparseable_objects)

            # Add errors for truly extra objects (not mentioned in description at all)
            if truly_extra_objects:
                result["error"] = (
                    "Model grid contains objects not mentioned in description"
                )
                result["extra_objects"] = sorted(truly_extra_objects)

            # Add info for missing objects
            if missing_objects:
                result["missing_objects"] = sorted(set(missing_objects))

        return result

    def evaluate_query(self, model_grid: str, query: str, reference_label: str) -> Dict:
        """Check if model grid produces correct answer to spatial query.

        Extracts objects and query type from query, infers their relationship from the grid,
        and compares to ground truth. Supports three query types:
        - full: Full spatial relationship (8 directions)
        - vertical: Vertical relationship only (above/below/same level)
        - horizontal: Horizontal relationship only (left/right/same column)

        Returns:
            Dict with keys: correct, query_objects, query_type, expected, actual.
            On error: correct=False, error, query_objects, expected.
        """
        obj1, obj2, query_type = self.parse_query(query)

        if obj1 is None or obj2 is None or query_type is None:
            return {
                "correct": False,
                "error": f"Failed to parse query: {query}",
                "query_objects": (obj1, obj2),
                "query_type": query_type,
                "expected": reference_label,
            }

        try:
            model_positions = self.parser.parse(model_grid)
        except ValueError as e:
            return {
                "correct": False,
                "error": f"Model grid parsing error: {str(e)}",
                "query_objects": (obj1, obj2),
                "query_type": query_type,
                "expected": reference_label,
            }

        if obj1 not in model_positions:
            return {
                "correct": False,
                "error": f'Object "{obj1}" not found in model grid',
                "query_objects": (obj1, obj2),
                "query_type": query_type,
                "expected": reference_label,
                "missing_object": obj1,
            }

        if obj2 not in model_positions:
            return {
                "correct": False,
                "error": f'Object "{obj2}" not found in model grid',
                "query_objects": (obj1, obj2),
                "query_type": query_type,
                "expected": reference_label,
                "missing_object": obj2,
            }

        model_normalized = normalize_positions(model_positions)

        # Infer relation based on query type
        if query_type == "vertical":
            inferred_relation = get_vertical_relation(
                model_normalized[obj1], model_normalized[obj2]
            )
        elif query_type == "horizontal":
            inferred_relation = get_horizontal_relation(
                model_normalized[obj1], model_normalized[obj2]
            )
        else:  # query_type == "full"
            model_relations = self.relation_extractor.extract_all_relations(
                model_normalized
            )
            inferred_relation = self._get_relation_from_graph(
                model_relations, obj1, obj2
            )

        is_correct = inferred_relation == reference_label.lower()

        return {
            "correct": is_correct,
            "query_objects": (obj1, obj2),
            "query_type": query_type,
            "expected": reference_label,
            "actual": inferred_relation if inferred_relation else "unknown",
        }


if __name__ == "__main__":
    # Simple CLI for quick testing
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate ASCII grid spatial relationships"
    )
    parser.add_argument("--reference", type=str, help="Reference ASCII grid file")
    parser.add_argument("--model", type=str, help="Model-generated ASCII grid file")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    # Query-based evaluation arguments
    parser.add_argument(
        "--query-relation",
        type=str,
        help='Query relation string (e.g., "Where is A relative to B?")',
    )
    parser.add_argument(
        "--label", type=str, help="Ground truth label for query-based evaluation"
    )

    # Description-based evaluation arguments
    parser.add_argument(
        "--description",
        type=str,
        help="Natural language description of spatial relationships",
    )

    args = parser.parse_args()

    # Description-based evaluation mode
    if args.description and args.model:
        with open(args.model, "r", encoding="utf-8") as f:
            model_grid = f.read()

        evaluator = ASCIIEvaluator()
        result = evaluator.evaluate_description(
            model_grid=model_grid, description=args.description
        )

        print("\nDescription-Based Evaluation Result:")
        print(f"  Description: {args.description}")
        print(f"  Correct: {result['correct']}")

        if "accuracy" in result:
            print(f"  Accuracy: {result['accuracy']:.2%}")
            print(f"  Total relations: {result['total_relations']}")
            print(
                f"  Correct: {result['correct_count']}, Incorrect: {result['incorrect_count']}"
            )

            if result.get("unparsed_count"):
                print(f"  Unparsed sentences: {result['unparsed_count']}")
                if result.get("unparseable_objects"):
                    print(
                        f"  Unparseable objects: {', '.join(result['unparseable_objects'])}"
                    )

            if result.get("incorrect_relations"):
                print("\n  Incorrect Relations:")
                for rel in result["incorrect_relations"]:
                    if "error" in rel:
                        print(f"    {rel['relation']}: {rel['error']}")
                    else:
                        print(
                            f"    {rel['relation']}: Expected={rel['expected']}, Actual={rel['actual']}"
                        )

            if result.get("missing_objects"):
                print(f"\n  Missing objects: {', '.join(result['missing_objects'])}")

        if "error" in result:
            print(f"\n  Error: {result['error']}")
            if result.get("extra_objects"):
                print(f"  Extra objects: {', '.join(result['extra_objects'])}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")

    # Query-based evaluation mode
    elif args.query_relation and args.label and args.model:
        with open(args.model, "r", encoding="utf-8") as f:
            model_grid = f.read()

        evaluator = ASCIIEvaluator()
        result = evaluator.evaluate_query(
            model_grid=model_grid, query=args.query_relation, reference_label=args.label
        )

        print("\nQuery-Based Evaluation Result:")
        print(f"  Query: {args.query_relation}")
        print(f"  Query Type: {result.get('query_type', 'N/A')}")
        print(f"  Query Objects: {result.get('query_objects', 'N/A')}")
        print(f"  Expected: {result.get('expected', 'N/A')}")
        print(f"  Actual: {result.get('actual', 'N/A')}")
        print(f"  Correct: {result['correct']}")

        if "error" in result:
            print(f"  Error: {result['error']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")

    elif args.reference and args.model:
        # Single evaluation
        with open(args.reference, "r", encoding="utf-8") as f:
            reference_grid = f.read()
        with open(args.model, "r", encoding="utf-8") as f:
            model_grid = f.read()

        evaluator = ASCIIEvaluator()
        result = evaluator.evaluate(reference_grid, model_grid)

        print("\nEvaluation Result:")
        print(f"  Correct: {result['correct']}")
        if "accuracy" in result:
            print(f"  Accuracy: {result['accuracy']:.2%}")
            print(f"  Total relations: {result['total_relations']}")
            print(f"  Correct relations: {result['correct_count']}")
            print(f"  Incorrect relations: {result['incorrect_count']}")

            if result["incorrect_relations"]:
                print("\n  Incorrect Relations:")
                for rel in result["incorrect_relations"]:
                    pair = rel["pair"]
                    print(
                        f"    {pair[0]} → {pair[1]}: "
                        f"Expected={rel['expected']}, Actual={rel['actual']}"
                    )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to {args.output}")

    else:
        parser.print_help()
        print("\nError: Provide one of the following:")
        print("  1. --reference and --model for grid comparison")
        print("  2. --query-relation, --label, and --model for query-based evaluation")
        print("  3. --description and --model for description-based evaluation")
