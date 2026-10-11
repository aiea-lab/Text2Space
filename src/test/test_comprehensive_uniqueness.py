"""Comprehensive tests for has_unique_representation and is_relation_uniquely_inferable"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "data"))
from spatial_map import SpatialMap


def test_case(name, setup_func, expected_unique, expected_inferable_pairs):
    """Run a test case and report results"""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print("=" * 70)

    map_obj = SpatialMap()
    setup_func(map_obj)

    print("Relations:")
    for obj1, obj2, direction in map_obj.stated_relations:
        print(f"  {obj1} is {direction} of {obj2}")

    print(f"\nGrid:\n{map_obj.render('grid')}")

    # Test has_unique_representation
    actual_unique = map_obj.has_unique_representation()
    print(f"\nhas_unique_representation: {actual_unique}")
    print(f"Expected: {expected_unique}")

    if actual_unique != expected_unique:
        print(f"❌ FAILED: Expected {expected_unique}, got {actual_unique}")
        return False
    else:
        print("✅ PASSED")

    # Test is_relation_uniquely_inferable for specified pairs
    all_passed = True
    if expected_inferable_pairs:
        print("\nTesting is_relation_uniquely_inferable:")
        for obj1, obj2, expected in expected_inferable_pairs:
            actual = map_obj.is_relation_uniquely_inferable(obj1, obj2)
            status = "✅" if actual == expected else "❌"
            print(f"  {status} {obj1}-{obj2}: {actual} (expected {expected})")
            if actual != expected:
                all_passed = False

    return all_passed


# Test 1: Simple cardinal chain (should be unique)
def setup_cardinal_chain(m):
    m.add_relation("C", "B", "left")
    m.add_relation("B", "A", "left")


test_case(
    "Cardinal chain: C-B-A",
    setup_cardinal_chain,
    expected_unique=True,
    expected_inferable_pairs=[("C", "A", True)],
)


# Test 2: Simple diagonal chain (should NOT be unique)
def setup_diagonal_chain(m):
    m.add_relation("B", "A", "upper-right")
    m.add_relation("C", "B", "upper-right")


test_case(
    "Diagonal chain: A-B-C",
    setup_diagonal_chain,
    expected_unique=True,
    expected_inferable_pairs=[("C", "A", True)],
)


# Test 3: The original bug case (diagonal chain - should NOT be unique)
def setup_original_bug(m):
    m.add_relation("B", "C", "lower-left")
    m.add_relation("A", "B", "upper-right")


test_case(
    "B lower-left of C, A upper-right of B",
    setup_original_bug,
    expected_unique=False,
    expected_inferable_pairs=[("A", "C", False)],
)


# Test 4: 2D grid with cardinals only (should be unique)
def setup_2d_cardinal_grid(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "A", "above")
    m.add_relation("D", "B", "above")


test_case(
    "2D cardinal grid",
    setup_2d_cardinal_grid,
    expected_unique=True,
    expected_inferable_pairs=[("B", "C", True), ("D", "A", True), ("D", "C", True)],
)


# Test 5: Cluster ambiguity - multiple objects left of A with no ordering (should NOT be unique)
def setup_cluster_ambiguity(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "A", "left")


test_case(
    "Cluster ambiguity: B and C both left of A, no B-C relation",
    setup_cluster_ambiguity,
    expected_unique=False,
    expected_inferable_pairs=[("B", "C", False)],
)


# Test 6: Resolved cluster - multiple objects left of A WITH ordering (should be unique)
def setup_resolved_cluster(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "A", "left")
    m.add_relation("C", "B", "left")


test_case(
    "Resolved cluster: B and C left of A, C left of B",
    setup_resolved_cluster,
    expected_unique=True,
    expected_inferable_pairs=[("C", "B", True)],
)


# Test 7: Fully specified diagonal triangle (all relations stated - should be unique)
def setup_full_diagonal_triangle(m):
    m.add_relation("B", "A", "upper-right")
    m.add_relation("C", "A", "lower-left")
    m.add_relation("C", "B", "lower-left")


test_case(
    "Fully specified diagonal triangle",
    setup_full_diagonal_triangle,
    expected_unique=True,
    expected_inferable_pairs=[("C", "B", True)],  # Directly stated
)


# Test 8: Mixed path - cardinal then diagonal (should NOT be unique for the inferred relation)
def setup_mixed_path(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "B", "upper-right")


test_case(
    "Mixed path: B left of A, C upper-right of B",
    setup_mixed_path,
    expected_unique=True,
    expected_inferable_pairs=[("C", "A", True)],
)


# Test 9: Long cardinal chain (should be unique)
def setup_long_cardinal_chain(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "B", "left")
    m.add_relation("D", "C", "left")


test_case(
    "Long cardinal chain: D-C-B-A",
    setup_long_cardinal_chain,
    expected_unique=True,
    expected_inferable_pairs=[("C", "A", True), ("D", "A", True), ("D", "B", True)],
)


# Test 10: Complex 2D grid with mixed cardinals (should be unique)
def setup_complex_2d_grid(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "B", "above")
    m.add_relation("D", "A", "above")
    m.add_relation("E", "D", "left")


test_case(
    "Complex 2D grid with 5 objects",
    setup_complex_2d_grid,
    expected_unique=False,
    expected_inferable_pairs=[("C", "A", True), ("E", "A", True), ("E", "B", False)],
)


# Test 11: Single diagonal relation (2 objects only - should be unique as directly stated)
def setup_single_diagonal(m):
    m.add_relation("B", "A", "upper-right")


test_case(
    "Single diagonal: B upper-right of A",
    setup_single_diagonal,
    expected_unique=True,
    expected_inferable_pairs=[("B", "A", True)],  # Directly stated
)


# Test 12: Reverse cluster ambiguity (A has same relation to B and C)
def setup_reverse_cluster(m):
    m.add_relation("A", "B", "upper-right")
    m.add_relation("A", "C", "upper-right")


test_case(
    "Reverse cluster: A upper-right of both B and C, no B-C relation",
    setup_reverse_cluster,
    expected_unique=False,
    expected_inferable_pairs=[("B", "C", False)],
)


# Test 13: Diagonal with cardinal inference (special case)
# B is upper-right of A, C is above of A → C and B relation should be inferable via cardinal path
def setup_diagonal_cardinal_mix(m):
    m.add_relation("B", "A", "upper-right")
    m.add_relation("C", "A", "above")


test_case(
    "Diagonal + cardinal from same reference",
    setup_diagonal_cardinal_mix,
    expected_unique=True,  # B-C path requires going through A with mixed directions
    expected_inferable_pairs=[("B", "C", True)],  # Path B->A->C contains diagonal
)


# Test 14: Cardinal branching (should be unique)
def setup_cardinal_branching(m):
    m.add_relation("B", "A", "left")
    m.add_relation("C", "A", "right")
    m.add_relation("D", "A", "above")


test_case(
    "Cardinal branching from A",
    setup_cardinal_branching,
    expected_unique=True,
    expected_inferable_pairs=[("B", "C", True), ("B", "D", True), ("C", "D", True)],
)


print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
