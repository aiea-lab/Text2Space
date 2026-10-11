"""
Test script for is_vertical_relation_uniquely_inferable and is_horizontal_relation_uniquely_inferable.

This demonstrates the difference between full relation uniqueness and component uniqueness.
"""

import sys
from pathlib import Path

# Add parent directory to path to import spatial_map
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from spatial_map import SpatialMap


def test_case_1_unique_full_relation():
    """Test Case 1: Full relation is unique (all components are unique)"""
    print("\n" + "=" * 70)
    print("TEST CASE 1: Unique Full Relation")
    print("=" * 70)

    spatial_map = SpatialMap()
    spatial_map.add_relation("A", "B", "above")
    spatial_map.add_relation("C", "B", "left")

    print("\nSetup:")
    print("- A is above B")
    print("- C is left of B")
    print("\nASCII visualization:")
    print(spatial_map.render("simple"))

    # Test A and C
    obj1, obj2 = "A", "C"
    print(f"\nTesting relation between {obj1} and {obj2}:")
    print(
        f"  Full relation unique: {spatial_map.is_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Vertical unique: {spatial_map.is_vertical_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Horizontal unique: {spatial_map.is_horizontal_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Actual relation: {obj1} is {spatial_map.get_relation(obj1, obj2)} of {obj2}"
    )


def test_case_2_vertical_ambiguous():
    """Test Case 2: Vertical component is ambiguous, but horizontal is unique"""
    print("\n" + "=" * 70)
    print("TEST CASE 2: Vertical Ambiguous, Horizontal Unique")
    print("=" * 70)

    spatial_map = SpatialMap()
    spatial_map.add_relation("B", "A", "upper-right")
    spatial_map.add_relation("C", "A", "lower-right")

    print("\nSetup:")
    print("- B is upper-right of A")
    print("- C is lower-right of A")
    print("- No relation between B and C")
    print("\nASCII visualization:")
    print(spatial_map.render("simple"))

    # Test B and C
    obj1, obj2 = "B", "C"
    print(f"\nTesting relation between {obj1} and {obj2}:")
    print(
        f"  Full relation unique: {spatial_map.is_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Vertical unique: {spatial_map.is_vertical_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Horizontal unique: {spatial_map.is_horizontal_relation_uniquely_inferable(obj1, obj2)}"
    )

    print(f"\nExplanation:")
    print(f"  - Full relation is NOT unique (could be anywhere vertically)")
    print(f"  - Vertical is unique: both paths show {obj1} is above {obj2}")
    print(f"  - Horizontal is unique: both are right of A, so same column")
    print(
        f"  Actual relation: {obj1} is {spatial_map.get_relation(obj1, obj2)} of {obj2}"
    )


def test_case_3_horizontal_ambiguous():
    """Test Case 3: Horizontal component is ambiguous, but vertical is unique"""
    print("\n" + "=" * 70)
    print("TEST CASE 3: Horizontal Ambiguous, Vertical Unique")
    print("=" * 70)

    spatial_map = SpatialMap()
    spatial_map.add_relation("B", "A", "upper-left")
    spatial_map.add_relation("C", "A", "upper-right")

    print("\nSetup:")
    print("- B is upper-left of A")
    print("- C is upper-right of A")
    print("- No relation between B and C")
    print("\nASCII visualization:")
    print(spatial_map.render("simple"))

    # Test B and C
    obj1, obj2 = "B", "C"
    print(f"\nTesting relation between {obj1} and {obj2}:")
    print(
        f"  Full relation unique: {spatial_map.is_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Vertical unique: {spatial_map.is_vertical_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Horizontal unique: {spatial_map.is_horizontal_relation_uniquely_inferable(obj1, obj2)}"
    )

    print(f"\nExplanation:")
    print(f"  - Full relation is NOT unique (could be anywhere horizontally)")
    print(f"  - Vertical is unique: both are above A, so same level")
    print(
        f"  - Horizontal is unique: {obj1} is left, {obj2} is right, so {obj1} is left of {obj2}"
    )
    print(
        f"  Actual relation: {obj1} is {spatial_map.get_relation(obj1, obj2)} of {obj2}"
    )


def test_case_4_both_components_ambiguous():
    """Test Case 4: Both components are ambiguous"""
    print("\n" + "=" * 70)
    print("TEST CASE 4: Both Components Ambiguous")
    print("=" * 70)

    spatial_map = SpatialMap()
    spatial_map.add_relation("B", "A", "upper-right")
    spatial_map.add_relation("C", "A", "upper-right")

    print("\nSetup:")
    print("- B is upper-right of A")
    print("- C is upper-right of A")
    print("- No relation between B and C")
    print("\nASCII visualization:")
    print(spatial_map.render("simple"))

    # Test B and C
    obj1, obj2 = "B", "C"
    print(f"\nTesting relation between {obj1} and {obj2}:")
    print(
        f"  Full relation unique: {spatial_map.is_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Vertical unique: {spatial_map.is_vertical_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Horizontal unique: {spatial_map.is_horizontal_relation_uniquely_inferable(obj1, obj2)}"
    )

    print(f"\nExplanation:")
    print(f"  - Full relation is NOT unique")
    print(
        f"  - Vertical is NOT unique (both upper-right of A, but relative vertical position unknown)"
    )
    print(
        f"  - Horizontal is NOT unique (both upper-right of A, but relative horizontal position unknown)"
    )
    print(
        f"  Actual relation: {obj1} is {spatial_map.get_relation(obj1, obj2)} of {obj2}"
    )


def test_case_5_complex_layout():
    """Test Case 5: More complex layout with multiple paths"""
    print("\n" + "=" * 70)
    print("TEST CASE 5: Complex Layout")
    print("=" * 70)

    spatial_map = SpatialMap()
    spatial_map.add_relation("B", "A", "above")
    spatial_map.add_relation("C", "A", "right")
    spatial_map.add_relation("D", "B", "right")
    spatial_map.add_relation("E", "C", "above")

    print("\nSetup:")
    print("- B is above A")
    print("- C is right of A")
    print("- D is right of B")
    print("- E is above C")
    print("\nASCII visualization:")
    print(spatial_map.render("simple"))

    # Test several pairs
    test_pairs = [("D", "A"), ("E", "B"), ("D", "E")]

    for obj1, obj2 in test_pairs:
        print(f"\nTesting relation between {obj1} and {obj2}:")
        print(
            f"  Full relation unique: {spatial_map.is_relation_uniquely_inferable(obj1, obj2)}"
        )
        print(
            f"  Vertical unique: {spatial_map.is_vertical_relation_uniquely_inferable(obj1, obj2)}"
        )
        print(
            f"  Horizontal unique: {spatial_map.is_horizontal_relation_uniquely_inferable(obj1, obj2)}"
        )
        print(
            f"  Actual relation: {obj1} is {spatial_map.get_relation(obj1, obj2)} of {obj2}"
        )


def test_case_6_directly_stated():
    """Test Case 6: Directly stated relations (always unique)"""
    print("\n" + "=" * 70)
    print("TEST CASE 6: Directly Stated Relations")
    print("=" * 70)

    spatial_map = SpatialMap()
    spatial_map.add_relation("A", "B", "upper-left")

    print("\nSetup:")
    print("- A is upper-left of B")
    print("\nASCII visualization:")
    print(spatial_map.render("simple"))

    obj1, obj2 = "A", "B"
    print(f"\nTesting relation between {obj1} and {obj2}:")
    print(
        f"  Full relation unique: {spatial_map.is_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Vertical unique: {spatial_map.is_vertical_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Horizontal unique: {spatial_map.is_horizontal_relation_uniquely_inferable(obj1, obj2)}"
    )
    print(
        f"  Actual relation: {obj1} is {spatial_map.get_relation(obj1, obj2)} of {obj2}"
    )
    print("\nExplanation: Directly stated relations are always uniquely inferable")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("TESTING VERTICAL AND HORIZONTAL RELATION UNIQUENESS")
    print("=" * 70)

    test_case_1_unique_full_relation()
    test_case_2_vertical_ambiguous()
    test_case_3_horizontal_ambiguous()
    test_case_4_both_components_ambiguous()
    test_case_5_complex_layout()
    test_case_6_directly_stated()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
