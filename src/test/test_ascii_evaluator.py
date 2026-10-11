#!/usr/bin/env python3
"""
Comprehensive Test Suite for ASCII Evaluator

Tests all evaluation modes with generated spatial instances:
1. Grid comparison mode (reference vs model grids)
2. Query-based evaluation mode
3. Description-based evaluation mode
4. Edge cases and error handling

Note: This test suite uses a fixed random seed (default: 42) for reproducibility.
All test results should be consistent across runs with the same seed value.
To use a different seed, modify the random_seed parameter when creating ASCIIEvaluatorTester.
"""

import json
import os
import random
import sys
from typing import Dict, List

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# Add src/data to path for spatial_map import
sys.path.insert(0, os.path.join(project_root, "src", "data"))

# Import modules
from src.data.spatial_data_generator import SpatialDataGenerator
from src.evaluation.ascii_evaluator import ASCIIEvaluator


class ASCIIEvaluatorTester:
    """Comprehensive tester for ASCII evaluator functionality."""

    def __init__(self, num_test_instances: int = 2000, random_seed: int = 42):
        """Initialize tester with configurable number of test instances.

        Args:
            num_test_instances: Number of test instances to generate
            random_seed: Random seed for reproducible test results (default: 42)
        """
        self.num_test_instances = num_test_instances
        self.random_seed = random_seed

        # Set random seed for reproducibility
        random.seed(random_seed)

        self.generator = SpatialDataGenerator(
            max_components=8, max_hops=12, min_components=2, min_hops=1
        )
        self.evaluator = ASCIIEvaluator()
        self.test_results = {
            "grid_comparison": [],
            "query_based": [],
            "query_based_incorrect": [],
            "query_based_all_types": [],
            "description_based": [],
            "description_mismatch": [],
            "verify_description": [],
            "edge_cases": [],
        }

    def generate_test_instances(self) -> List[Dict]:
        """Generate test instances using SpatialDataGenerator."""
        print(f"Generating {self.num_test_instances} test instances...")
        instances = []
        attempts = 0
        max_attempts = self.num_test_instances * 3

        while len(instances) < self.num_test_instances and attempts < max_attempts:
            attempts += 1
            instance = self.generator.generate_instance()
            if instance is not None:
                instance["id"] = f"test-{len(instances) + 1}"
                instances.append(instance)
                if len(instances) % 10 == 0:
                    print(
                        f"  Generated {len(instances)}/{self.num_test_instances} instances"
                    )

        print(f"Successfully generated {len(instances)} test instances\n")
        return instances

    def test_grid_comparison_mode(self, instances: List[Dict]):
        """Test grid comparison mode with all three ASCII formats."""
        print("=" * 80)
        print("TEST 1: Grid Comparison Mode (Same Format)")
        print("=" * 80)

        formats = ["simple", "grid", "panel"]

        for format_type in formats:
            print(f"\nTesting {format_type} format...")
            correct_count = 0
            total_count = 0

            for instance in instances[:100]:  # Test first 100 instances per format
                reference_grid = instance["ascii"][format_type]
                model_grid = instance["ascii"][
                    format_type
                ]  # Same grid (should be 100% correct)

                result = self.evaluator.evaluate_grids(reference_grid, model_grid)
                total_count += 1

                if result["correct"]:
                    correct_count += 1
                else:
                    print(
                        f"  ❌ Instance {instance['id']} failed: {result.get('error', 'Unknown error')}"
                    )

                self.test_results["grid_comparison"].append(
                    {
                        "instance_id": instance["id"],
                        "format": format_type,
                        "result": result,
                    }
                )

            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(
                f"  ✓ {format_type} format: {correct_count}/{total_count} correct ({accuracy:.1f}%)"
            )

        print("\n✓ Grid comparison mode testing complete\n")

    def test_cross_format_comparison(self, instances: List[Dict]):
        """Test grid comparison across different ASCII formats."""
        print("=" * 80)
        print("TEST 1B: Cross-Format Grid Comparison")
        print("=" * 80)
        print(
            "Testing evaluator's ability to compare grids in different ASCII formats.\n"
        )

        formats = ["simple", "grid", "panel"]

        # Generate all cross-format pairs (e.g., simple vs grid, simple vs panel, etc.)
        format_pairs = []
        for i, ref_format in enumerate(formats):
            for model_format in formats[i + 1 :]:
                format_pairs.append((ref_format, model_format))

        for ref_format, model_format in format_pairs:
            print(f"\nTesting {ref_format} (reference) vs {model_format} (model)...")
            correct_count = 0
            total_count = 0

            for instance in instances[:100]:  # Test first 100 instances per pair
                reference_grid = instance["ascii"][ref_format]
                model_grid = instance["ascii"][model_format]

                result = self.evaluator.evaluate_grids(reference_grid, model_grid)
                total_count += 1

                if result["correct"]:
                    correct_count += 1
                else:
                    print(
                        f"  ❌ Instance {instance['id']} failed: {result.get('error', 'Unknown error')}"
                    )
                    if "incorrect_count" in result:
                        print(
                            f"     Incorrect relations: {result['incorrect_count']}/{result.get('total_relations', 0)}"
                        )

                self.test_results["grid_comparison"].append(
                    {
                        "instance_id": instance["id"],
                        "reference_format": ref_format,
                        "model_format": model_format,
                        "result": result,
                    }
                )

            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(
                f"  ✓ {ref_format} vs {model_format}: {correct_count}/{total_count} correct ({accuracy:.1f}%)"
            )

        print("\n✓ Cross-format grid comparison testing complete\n")

    def test_query_based_evaluation(self, instances: List[Dict]):
        """Test query-based evaluation mode."""
        print("=" * 80)
        print("TEST 2: Query-Based Evaluation Mode")
        print("=" * 80)

        formats = ["simple", "grid", "panel"]

        for format_type in formats:
            print(f"\nTesting {format_type} format...")
            correct_count = 0
            total_count = 0

            for instance in instances[:200]:  # Test first 200 instances per format
                model_grid = instance["ascii"][format_type]
                query_relation = instance["query_relation"]
                ground_truth = instance["label"]

                result = self.evaluator.evaluate_query(
                    model_grid=model_grid,
                    query=query_relation,
                    reference_label=ground_truth,
                )

                total_count += 1
                if result["correct"]:
                    correct_count += 1
                else:
                    print(
                        f"  ❌ Instance {instance['id']}: Expected '{result.get('expected')}', Got '{result.get('actual')}'"
                    )
                    if "error" in result:
                        print(f"     Error: {result['error']}")

                self.test_results["query_based"].append(
                    {
                        "instance_id": instance["id"],
                        "format": format_type,
                        "result": result,
                    }
                )

            accuracy = correct_count / total_count * 100 if total_count > 0 else 0
            print(
                f"  ✓ {format_type} format: {correct_count}/{total_count} correct ({accuracy:.1f}%)"
            )

        print("\n✓ Query-based evaluation mode testing complete\n")

    def test_query_based_incorrect_labels(self, instances: List[Dict]):
        """Test query-based evaluation with deliberately incorrect labels."""
        print("=" * 80)
        print("TEST 3: Query-Based Evaluation with Incorrect Labels")
        print("=" * 80)
        print("Testing evaluator's ability to detect incorrect ground truth labels.\n")

        # All 8 possible directions for full spatial queries
        all_directions = [
            "above",
            "below",
            "left",
            "right",
            "upper-left",
            "upper-right",
            "lower-left",
            "lower-right",
        ]

        formats = ["simple", "grid", "panel"]

        for format_type in formats:
            print(f"\nTesting {format_type} format...")
            total_tests = 0
            correctly_rejected = 0

            # Test 200 instances per format
            for instance in instances[:200]:
                if instance["query_type"] != "full":
                    continue

                model_grid = instance["ascii"][format_type]
                query_relation = instance["query_relation"]
                correct_label = instance["label"]

                # Test with each incorrect direction
                incorrect_directions = [d for d in all_directions if d != correct_label]

                for incorrect_label in incorrect_directions:
                    result = self.evaluator.evaluate_query(
                        model_grid=model_grid,
                        query=query_relation,
                        reference_label=incorrect_label,
                    )

                    total_tests += 1

                    # Should return False (incorrect label)
                    if not result["correct"]:
                        correctly_rejected += 1
                    else:
                        print(
                            f"  ⚠️  Instance {instance['id']}: Failed to reject incorrect label '{incorrect_label}' "
                            f"(actual: '{correct_label}')"
                        )

                    self.test_results["query_based_incorrect"].append(
                        {
                            "instance_id": instance["id"],
                            "format": format_type,
                            "correct_label": correct_label,
                            "tested_label": incorrect_label,
                            "result": result,
                            "correctly_rejected": not result["correct"],
                        }
                    )

            rejection_rate = (
                correctly_rejected / total_tests * 100 if total_tests > 0 else 0
            )
            print(
                f"  ✓ {format_type} format: {correctly_rejected}/{total_tests} "
                f"incorrect labels rejected ({rejection_rate:.1f}%)"
            )

        print("\n✓ Query-based incorrect label testing complete\n")

    def test_all_query_types(self, instances: List[Dict]):
        """Test all three query types: full, vertical, and horizontal."""
        print("=" * 80)
        print("TEST 4: Comprehensive Query Type Testing")
        print("=" * 80)
        print("Testing all query types: full spatial, vertical, and horizontal.\n")

        # Group instances by query type
        query_type_instances = {"full": [], "vertical": [], "horizontal": []}
        for instance in instances:
            query_type = instance.get("query_type", "full")
            if query_type in query_type_instances:
                query_type_instances[query_type].append(instance)

        formats = ["simple", "grid", "panel"]

        for query_type in ["full", "vertical", "horizontal"]:
            instances_for_type = query_type_instances[query_type]

            if not instances_for_type:
                print(
                    f"\n⚠️  No instances found for query type '{query_type}'. Skipping."
                )
                continue

            print(f"\n{'='*40}")
            print(f"Testing Query Type: {query_type.upper()}")
            print(f"{'='*40}")
            print(f"Available instances: {len(instances_for_type)}\n")

            for format_type in formats:
                correct_count = 0
                total_count = 0

                # Test up to 100 instances per format per query type
                test_instances = instances_for_type[: min(100, len(instances_for_type))]

                for instance in test_instances:
                    model_grid = instance["ascii"][format_type]
                    query_relation = instance["query_relation"]
                    ground_truth = instance["label"]

                    result = self.evaluator.evaluate_query(
                        model_grid=model_grid,
                        query=query_relation,
                        reference_label=ground_truth,
                    )

                    total_count += 1
                    if result["correct"]:
                        correct_count += 1
                    else:
                        print(
                            f"  ❌ {format_type} - Instance {instance['id']}: "
                            f"Expected '{result.get('expected')}', Got '{result.get('actual')}'"
                        )

                    self.test_results["query_based_all_types"].append(
                        {
                            "instance_id": instance["id"],
                            "query_type": query_type,
                            "format": format_type,
                            "result": result,
                        }
                    )

                accuracy = correct_count / total_count * 100 if total_count > 0 else 0
                print(
                    f"  ✓ {format_type} format: {correct_count}/{total_count} correct ({accuracy:.1f}%)"
                )

        print("\n✓ All query type testing complete\n")

    def test_description_based_evaluation(self, instances: List[Dict]):
        """Test description-based evaluation mode."""
        print("=" * 80)
        print("TEST 5: Description-Based Evaluation Mode")
        print("=" * 80)

        formats = ["simple", "grid", "panel"]

        for format_type in formats:
            print(f"\nTesting {format_type} format...")
            correct_count = 0
            total_count = 0
            total_relations = 0
            correct_relations = 0

            for instance in instances[:200]:  # Test first 200 instances per format
                model_grid = instance["ascii"][format_type]
                description = instance["description"]

                result = self.evaluator.evaluate_description(
                    model_grid=model_grid, description=description
                )

                total_count += 1
                if result["correct"]:
                    correct_count += 1
                else:
                    print(
                        f"  ❌ Instance {instance['id']}: {result.get('error', 'Relations mismatch')}"
                    )
                    if (
                        "incorrect_relations" in result
                        and result["incorrect_relations"]
                    ):
                        for rel in result["incorrect_relations"][
                            :3
                        ]:  # Show first 3 errors
                            if "error" in rel:
                                print(
                                    f"     - {rel.get('relation', 'Unknown')}: {rel['error']}"
                                )
                            else:
                                print(
                                    f"     - {rel.get('relation', 'Unknown')}: Expected '{rel.get('expected')}', Got '{rel.get('actual')}'"
                                )

                if "total_relations" in result:
                    total_relations += result["total_relations"]
                    correct_relations += result["correct_count"]

                self.test_results["description_based"].append(
                    {
                        "instance_id": instance["id"],
                        "format": format_type,
                        "result": result,
                    }
                )

            instance_accuracy = (
                correct_count / total_count * 100 if total_count > 0 else 0
            )
            relation_accuracy = (
                correct_relations / total_relations * 100 if total_relations > 0 else 0
            )
            print(f"  ✓ {format_type} format:")
            print(
                f"    - Instances: {correct_count}/{total_count} correct ({instance_accuracy:.1f}%)"
            )
            print(
                f"    - Relations: {correct_relations}/{total_relations} correct ({relation_accuracy:.1f}%)"
            )

        print("\n✓ Description-based evaluation mode testing complete\n")

    def test_description_mismatch(self, instances: List[Dict]):
        """Test description-based evaluation with mismatched descriptions and grids."""
        print("=" * 80)
        print("TEST 6: Description-Based Evaluation with Mismatches")
        print("=" * 80)
        print("Testing evaluator's ability to detect incorrect spatial relationships")
        print(
            "by using descriptions from one instance with ASCII grids from another.\n"
        )

        if len(instances) < 400:
            print("Not enough instances for mismatch testing. Skipping.\n")
            return

        formats = ["simple", "grid", "panel"]

        for format_type in formats:
            print(f"\nTesting {format_type} format...")
            total_count = 0
            should_fail_count = 0
            correctly_detected_count = 0

            # Test 200 mismatched pairs per format
            for i in range(200):
                # Use description from instance i, but ASCII from instance i+200
                desc_instance = instances[i]
                ascii_instance = instances[i + 200]

                description = desc_instance["description"]
                model_grid = ascii_instance["ascii"][format_type]

                result = self.evaluator.evaluate_description(
                    model_grid=model_grid, description=description
                )

                total_count += 1
                should_fail_count += 1

                # This SHOULD fail (description doesn't match grid)
                if not result["correct"]:
                    correctly_detected_count += 1
                else:
                    print(
                        f"  ⚠️  Instance pair ({desc_instance['id']}, {ascii_instance['id']}): "
                        f"Mismatch not detected (all relations matched unexpectedly)"
                    )

                self.test_results["description_mismatch"].append(
                    {
                        "description_instance_id": desc_instance["id"],
                        "ascii_instance_id": ascii_instance["id"],
                        "format": format_type,
                        "result": result,
                        "correctly_detected_mismatch": not result["correct"],
                    }
                )

            detection_rate = (
                correctly_detected_count / should_fail_count * 100
                if should_fail_count > 0
                else 0
            )
            print(
                f"  ✓ {format_type} format: {correctly_detected_count}/{should_fail_count} "
                f"mismatches correctly detected ({detection_rate:.1f}%)"
            )

        print("\n✓ Description mismatch testing complete\n")

    def test_verify_description_mode(self, instances: List[Dict]):
        """Test verify_description mode for stated relation validation."""
        print("=" * 80)
        print("TEST 6b: Verify Description Mode (Stated Relation Validation)")
        print("=" * 80)
        print(
            "Testing the verify_description mode which validates that stated relations"
        )
        print("in the description match the spatial relationships in the grid.\n")

        if not instances:
            print("No instances available for verify_description testing. Skipping.\n")
            return

        # Use first 100 instances for testing
        test_instances = instances[:100]

        # Test 1: Complete descriptions (should all pass)
        print("1. Testing complete descriptions...")
        correct_count = 0
        total_count = 0

        for instance in test_instances[:50]:
            model_grid = instance["ascii"]["simple"]
            description = instance["description"]

            result = self.evaluator.evaluate_description(
                model_grid=model_grid,
                description=description,
                mode="verify_description",
            )

            total_count += 1
            if result["correct"] and result["incorrect_count"] == 0:
                correct_count += 1
            else:
                # Show first few failures for debugging
                if total_count <= 5 and not result["correct"]:
                    print(
                        f"  ❌ Instance {instance['id']}: "
                        f"incorrect_count={result['incorrect_count']}, "
                        f"accuracy={result['accuracy']:.2f}"
                    )

        accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        print(f"  ✓ {correct_count}/{total_count} instances correct ({accuracy:.1f}%)")

        # Test 2: Partial descriptions (fewer relations)
        print("\n2. Testing partial descriptions (fewer relations stated)...")
        partial_count = 0
        total_partial = 0

        for instance in test_instances[50:80]:
            model_grid = instance["ascii"]["simple"]
            description = instance["description"]

            # Remove half the sentences to create partial description
            sentences = [s.strip() for s in description.split(".") if s.strip()]
            if len(sentences) >= 2:
                partial_description = ". ".join(sentences[: len(sentences) // 2]) + "."
            else:
                continue

            result = self.evaluator.evaluate_description(
                model_grid=model_grid,
                description=partial_description,
                mode="verify_description",
            )

            total_partial += 1
            # Partial descriptions should have fewer total_relations
            if result["total_relations"] < len(sentences):
                partial_count += 1

        detection_rate = partial_count / total_partial * 100 if total_partial > 0 else 0
        print(
            f"  ✓ {partial_count}/{total_partial} partial descriptions correctly have fewer "
            f"total_relations ({detection_rate:.1f}%)"
        )

        # Test 3: Correct stated relations validation
        print("\n3. Testing stated relations validation...")
        # Create a simple test case
        test_grid = "A\nB   C"  # A above B, B left of C
        test_desc_complete = "A is above B. B is left of C. A is upper-left of C."
        test_desc_minimal = "A is above B. B is left of C."
        test_desc_partial = "A is above B."  # Only 1 relation stated

        result_complete = self.evaluator.evaluate_description(
            test_grid, test_desc_complete, mode="verify_description"
        )
        result_minimal = self.evaluator.evaluate_description(
            test_grid, test_desc_minimal, mode="verify_description"
        )
        result_partial = self.evaluator.evaluate_description(
            test_grid, test_desc_partial, mode="verify_description"
        )

        complete_pass = (
            result_complete["correct"] and result_complete["incorrect_count"] == 0
        )
        minimal_pass = (
            result_minimal["correct"] and result_minimal["incorrect_count"] == 0
        )
        partial_pass = (
            result_partial["correct"] and result_partial["total_relations"] == 1
        )

        print(
            f"  {'✓' if complete_pass else '✗'} Complete description (3 relations): "
            f"correct={result_complete['correct']}, incorrect_count={result_complete['incorrect_count']}"
        )
        print(
            f"  {'✓' if minimal_pass else '✗'} Minimal description (2 relations): "
            f"correct={result_minimal['correct']}, incorrect_count={result_minimal['incorrect_count']}"
        )
        print(
            f"  {'✓' if partial_pass else '✗'} Partial description (1 relation): "
            f"correct={result_partial['correct']}, total_relations={result_partial['total_relations']}"
        )

        # Test 4: Accuracy calculation (no penalty for missing relations)
        print("\n4. Testing accuracy calculation...")
        # Complete: 3 correct / 3 = 1.0
        # Minimal: 2 correct / 2 = 1.0
        # Partial: 1 correct / 1 = 1.0 (no penalty for not stating other relations)
        complete_acc_ok = result_complete["accuracy"] == 1.0
        minimal_acc_ok = result_minimal["accuracy"] == 1.0
        partial_acc_ok = (
            result_partial["accuracy"] == 1.0
        )  # All stated relations are correct

        print(
            f"  {'✓' if complete_acc_ok else '✗'} Complete: accuracy={result_complete['accuracy']:.2f} (expected 1.0)"
        )
        print(
            f"  {'✓' if minimal_acc_ok else '✗'} Minimal: accuracy={result_minimal['accuracy']:.2f} (expected 1.0)"
        )
        print(
            f"  {'✓' if partial_acc_ok else '✗'} Partial: accuracy={result_partial['accuracy']:.2f} (expected 1.0)"
        )

        # Test 5: Backward compatibility (verify_ascii mode)
        print("\n5. Testing backward compatibility (verify_ascii mode)...")
        for instance in test_instances[:5]:
            model_grid = instance["ascii"]["simple"]
            description = instance["description"]

            # Default mode should be verify_ascii
            result_default = self.evaluator.evaluate_description(
                model_grid=model_grid, description=description
            )
            result_explicit = self.evaluator.evaluate_description(
                model_grid=model_grid, description=description, mode="verify_ascii"
            )

            # Should have 'correct_count' key (not 'covered_count')
            if "correct_count" in result_default and "correct_count" in result_explicit:
                if result_default["correct_count"] == result_explicit["correct_count"]:
                    continue
            print(f"  ✗ Instance {instance['id']}: backward compatibility issue")
            break
        else:
            print(f"  ✓ Backward compatibility maintained (5/5 instances)")

        print("\n✓ Verify description mode testing complete\n")

    def test_edge_cases(self, instances: List[Dict]):
        """Test edge cases and error handling."""
        print("=" * 80)
        print("TEST 7: Edge Cases and Error Handling")
        print("=" * 80)

        if not instances:
            print("No instances available for edge case testing")
            return

        instance = instances[0]
        reference_grid = instance["ascii"]["simple"]

        # Test 1: Empty grid
        print("\n1. Testing empty grid...")
        result = self.evaluator.evaluate_grids(reference_grid, "")
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected empty grid: {result.get('error', 'No error')}"
        )
        self.test_results["edge_cases"].append({"test": "empty_grid", "result": result})

        # Test 2: Duplicate objects in grid
        print("\n2. Testing duplicate objects...")
        duplicate_grid = "A   B\n\nA   C"  # A appears twice
        result = self.evaluator.evaluate_grids(reference_grid, duplicate_grid)
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected duplicates: {result.get('error', 'No error')}"
        )
        self.test_results["edge_cases"].append(
            {"test": "duplicate_objects", "result": result}
        )

        # Test 3: Missing objects
        print("\n3. Testing missing objects...")
        # Create a grid with fewer objects than reference
        partial_grid = "A   B"  # Missing some objects
        result = self.evaluator.evaluate_grids(reference_grid, partial_grid)
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected missing objects: {result.get('error', 'No error')}"
        )
        if "missing_objects" in result:
            print(f"      Missing: {result['missing_objects']}")
        self.test_results["edge_cases"].append(
            {"test": "missing_objects", "result": result}
        )

        # Test 4: Extra objects
        print("\n4. Testing extra objects...")
        extra_grid = reference_grid + "\nZ   Y   X"  # Add extra objects
        result = self.evaluator.evaluate_grids(reference_grid, extra_grid)
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected extra objects: {result.get('error', 'No error')}"
        )
        if "extra_objects" in result:
            print(f"      Extra: {result['extra_objects']}")
        self.test_results["edge_cases"].append(
            {"test": "extra_objects", "result": result}
        )

        # Test 5: Incorrect spatial relationships
        print("\n5. Testing incorrect spatial relationships...")
        # Swap two objects to create wrong relationships
        lines = reference_grid.strip().split("\n")
        if len(lines) > 0 and len(lines[0]) > 5:
            # Simple swap - reverse the first line
            wrong_grid = lines[0][::-1]
            if len(lines) > 1:
                wrong_grid += "\n" + "\n".join(lines[1:])
            result = self.evaluator.evaluate_grids(reference_grid, wrong_grid)
            print(
                f"   {'✓' if not result['correct'] else '✗'} Correctly detected wrong relationships"
            )
            if "incorrect_count" in result:
                print(
                    f"      Incorrect relations: {result['incorrect_count']}/{result.get('total_relations', 0)}"
                )
            self.test_results["edge_cases"].append(
                {"test": "wrong_relationships", "result": result}
            )

        # Test 6: Invalid query format
        print("\n6. Testing invalid query format...")
        result = self.evaluator.evaluate_query(
            model_grid=reference_grid,
            query="This is not a valid query",
            reference_label="above",
        )
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected invalid query: {result.get('error', 'No error')}"
        )
        self.test_results["edge_cases"].append(
            {"test": "invalid_query", "result": result}
        )

        # Test 7: Query with missing objects
        print("\n7. Testing query with missing objects...")
        result = self.evaluator.evaluate_query(
            model_grid="A   B",
            query="Where is Z relative to Y?",
            reference_label="above",
        )
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected missing query objects: {result.get('error', 'No error')}"
        )
        self.test_results["edge_cases"].append(
            {"test": "missing_query_objects", "result": result}
        )

        # Test 8: Empty description
        print("\n8. Testing empty description...")
        result = self.evaluator.evaluate_description(
            model_grid=reference_grid, description=""
        )
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected empty description: {result.get('error', 'No error')}"
        )
        self.test_results["edge_cases"].append(
            {"test": "empty_description", "result": result}
        )

        # Test 9: Description with unparseable relations
        print("\n9. Testing unparseable description...")
        result = self.evaluator.evaluate_description(
            model_grid=reference_grid,
            description="This is just random text with no spatial relations.",
        )
        print(
            f"   {'✓' if not result['correct'] else '✗'} Correctly detected unparseable description: {result.get('error', 'No error')}"
        )
        self.test_results["edge_cases"].append(
            {"test": "unparseable_description", "result": result}
        )

        print("\n✓ Edge case testing complete\n")

    def generate_report(self, instances: List[Dict]) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("ASCII EVALUATOR COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"\nTotal test instances generated: {len(instances)}\n")

        # Summary statistics
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)

        # Grid comparison summary
        grid_comp_results = self.test_results["grid_comparison"]
        if grid_comp_results:
            total = len(grid_comp_results)
            correct = sum(1 for r in grid_comp_results if r["result"]["correct"])
            report.append(f"\n1. Grid Comparison Mode:")
            report.append(f"   Total tests: {total}")
            report.append(f"   Correct: {correct} ({correct/total*100:.1f}%)")
            report.append(
                f"   Failed: {total - correct} ({(total-correct)/total*100:.1f}%)"
            )

        # Query-based summary
        query_results = self.test_results["query_based"]
        if query_results:
            total = len(query_results)
            correct = sum(1 for r in query_results if r["result"]["correct"])
            report.append(f"\n2. Query-Based Evaluation (Correct Labels):")
            report.append(f"   Total tests: {total}")
            report.append(f"   Correct: {correct} ({correct/total*100:.1f}%)")
            report.append(
                f"   Failed: {total - correct} ({(total-correct)/total*100:.1f}%)"
            )

        # Query-based incorrect labels summary
        query_incorrect_results = self.test_results["query_based_incorrect"]
        if query_incorrect_results:
            total = len(query_incorrect_results)
            correctly_rejected = sum(
                1 for r in query_incorrect_results if r["correctly_rejected"]
            )
            report.append(f"\n3. Query-Based Evaluation (Incorrect Labels):")
            report.append(f"   Total tests: {total}")
            report.append(
                f"   Correctly rejected: {correctly_rejected} ({correctly_rejected/total*100:.1f}%)"
            )
            report.append(
                f"   Incorrectly accepted: {total - correctly_rejected} ({(total-correctly_rejected)/total*100:.1f}%)"
            )

        # All query types summary
        query_types_results = self.test_results["query_based_all_types"]
        if query_types_results:
            report.append(f"\n4. Comprehensive Query Type Testing:")
            for qtype in ["full", "vertical", "horizontal"]:
                type_results = [
                    r for r in query_types_results if r["query_type"] == qtype
                ]
                if type_results:
                    total = len(type_results)
                    correct = sum(1 for r in type_results if r["result"]["correct"])
                    report.append(
                        f"   {qtype.capitalize()}: {correct}/{total} correct ({correct/total*100:.1f}%)"
                    )

        # Description-based summary
        desc_results = self.test_results["description_based"]
        if desc_results:
            total = len(desc_results)
            correct = sum(1 for r in desc_results if r["result"]["correct"])
            total_relations = sum(
                r["result"].get("total_relations", 0) for r in desc_results
            )
            correct_relations = sum(
                r["result"].get("correct_count", 0) for r in desc_results
            )
            report.append(f"\n5. Description-Based Evaluation (Matching Data):")
            report.append(f"   Total tests: {total}")
            report.append(f"   Instances correct: {correct} ({correct/total*100:.1f}%)")
            report.append(f"   Total relations: {total_relations}")
            report.append(
                f"   Relations correct: {correct_relations} ({correct_relations/total_relations*100:.1f}%)"
            )

        # Description mismatch summary
        desc_mismatch_results = self.test_results["description_mismatch"]
        if desc_mismatch_results:
            total = len(desc_mismatch_results)
            correctly_detected = sum(
                1 for r in desc_mismatch_results if r["correctly_detected_mismatch"]
            )
            report.append(f"\n6. Description-Based Evaluation (Mismatched Data):")
            report.append(f"   Total tests: {total}")
            report.append(
                f"   Mismatches detected: {correctly_detected} ({correctly_detected/total*100:.1f}%)"
            )
            report.append(
                f"   Mismatches not detected: {total - correctly_detected} ({(total-correctly_detected)/total*100:.1f}%)"
            )

        # Edge cases summary
        edge_results = self.test_results["edge_cases"]
        if edge_results:
            report.append(f"\n7. Edge Cases:")
            report.append(f"   Total edge case tests: {len(edge_results)}")
            for edge_test in edge_results:
                test_name = edge_test["test"].replace("_", " ").title()
                status = (
                    "✓ Passed" if not edge_test["result"]["correct"] else "✗ Failed"
                )
                report.append(f"   - {test_name}: {status}")

        # Dataset characteristics
        report.append("\n" + "=" * 80)
        report.append("TEST DATASET CHARACTERISTICS")
        report.append("=" * 80)

        if instances:
            num_elements = [inst["num_components"] for inst in instances]
            num_hops = [inst["num_relations"] for inst in instances]
            direct_count = sum(
                1 for inst in instances if inst.get("is_directly_stated", False)
            )
            clock_count = sum(
                1 for inst in instances if inst.get("description_mode") == "clock"
            )
            base_only_count = sum(
                1 for inst in instances if inst.get("description_mode") == "spatial"
            )
            unique_count = sum(
                1 for inst in instances if inst.get("has_unique_layout", True)
            )

            report.append(
                f"\n  Components: min={min(num_elements)}, max={max(num_elements)}, avg={sum(num_elements)/len(num_elements):.2f}"
            )
            report.append(
                f"  Hops: min={min(num_hops)}, max={max(num_hops)}, avg={sum(num_hops)/len(num_hops):.2f}"
            )
            report.append(
                f"  Direct relations: {direct_count} ({direct_count/len(instances)*100:.1f}%)"
            )
            report.append(
                f"  Indirect relations: {len(instances) - direct_count} ({(len(instances)-direct_count)/len(instances)*100:.1f}%)"
            )
            report.append(
                f"  Clock mode: {clock_count} ({clock_count/len(instances)*100:.1f}%)"
            )
            report.append(
                f"  Base only mode: {base_only_count} ({base_only_count/len(instances)*100:.1f}%)"
            )
            report.append(
                f"  Unique representations: {unique_count} ({unique_count/len(instances)*100:.1f}%)"
            )

        report.append("\n" + "=" * 80)
        report.append("TEST COMPLETE")
        report.append("=" * 80)

        return "\n".join(report)

    def run_all_tests(self):
        """Run all tests and generate report."""
        print("\n" + "=" * 80)
        print("STARTING COMPREHENSIVE ASCII EVALUATOR TEST SUITE")
        print("=" * 80 + "\n")

        # Generate test instances
        instances = self.generate_test_instances()

        if not instances:
            print("❌ Failed to generate test instances. Aborting.")
            return

        # Run all test suites
        self.test_grid_comparison_mode(instances)
        self.test_cross_format_comparison(instances)
        self.test_query_based_evaluation(instances)
        self.test_query_based_incorrect_labels(instances)
        self.test_all_query_types(instances)
        self.test_description_based_evaluation(instances)
        self.test_description_mismatch(instances)
        self.test_verify_description_mode(instances)
        self.test_edge_cases(instances)

        # Generate and print report
        report = self.generate_report(instances)
        print(report)

        # Save results to file
        output_file = "src/test/ascii_evaluator_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "test_results": self.test_results,
                    "test_instances": instances[
                        :5
                    ],  # Save first 5 instances as examples
                    "summary": {
                        "total_instances": len(instances),
                        "tests_run": sum(len(v) for v in self.test_results.values()),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n✓ Detailed results saved to {output_file}")

        # Save report to text file
        report_file = "src/test/ascii_evaluator_test_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"✓ Test report saved to {report_file}\n")


def main():
    """Main entry point for test suite."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive ASCII evaluator tests"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=2000,
        help="Number of test instances to generate (default: 2000)",
    )
    args = parser.parse_args()

    print(f"Using random seed: {args.seed}")
    print(f"Generating {args.num_instances} test instances\n")

    tester = ASCIIEvaluatorTester(
        num_test_instances=args.num_instances, random_seed=args.seed
    )
    tester.run_all_tests()


if __name__ == "__main__":
    main()
