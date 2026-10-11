import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "data"))
from spatial_map import SpatialMap
from spatial_utils import infer_relation_from_path


def test_1_hop():
    """Direct relation: single fact, no 'Since' clause."""
    m = SpatialMap()
    m.add_relation("B", "A", "above")

    result = m.get_reasoning_steps("B", "A")
    assert result == "B is above A.", f"Got: {result}"
    print("  PASSED: 1-hop direct")


def test_2_hop():
    """Two hops: two facts, no 'Since' clause (composition would reveal the answer)."""
    m = SpatialMap()
    m.add_relation("B", "A", "above")
    m.add_relation("C", "B", "right")

    result = m.get_reasoning_steps("C", "A")
    assert result == "B is above A. C is to the right of B.", f"Got: {result}"

    # "Since" should NOT appear (only 2 hops)
    assert "Since" not in result
    print("  PASSED: 2-hop no composition")


def test_3_hop():
    """Three hops: two facts + 'Since' intermediate + third fact."""
    m = SpatialMap()
    m.add_relation("B", "A", "above")
    m.add_relation("C", "B", "right")
    m.add_relation("D", "C", "below")

    result = m.get_reasoning_steps("D", "A")
    assert "Since B is above A and C is to the right of B" in result
    assert "C is above and to the right of A" in result
    assert result.endswith("D is below C.")
    print("  PASSED: 3-hop with intermediate composition")


def test_4_hop():
    """Four hops: two 'Since' clauses, last fact without composition."""
    m = SpatialMap()
    m.add_relation("B", "A", "above")
    m.add_relation("C", "B", "right")
    m.add_relation("D", "C", "below")
    m.add_relation("E", "D", "left")

    result = m.get_reasoning_steps("E", "A")
    # Should have two "Since" clauses
    assert (
        result.count("Since") == 2
    ), f"Expected 2 'Since', got {result.count('Since')}"
    # Last fact should be the ending
    assert result.endswith("E is to the left of D.")
    print("  PASSED: 4-hop with two compositions")


def test_consistency():
    """The path used for reasoning must produce the same relation as get_relation."""
    m = SpatialMap()
    m.add_relation("B", "A", "above")
    m.add_relation("C", "B", "right")
    m.add_relation("D", "C", "below")

    # Get the path directly
    paths, node_seqs = m._find_all_shortest_paths("A", "D", return_nodes=True)
    assert len(paths) > 0
    composed = infer_relation_from_path(paths[0])
    actual = m.get_relation("D", "A")
    assert composed == actual, f"Path gives {composed}, get_relation gives {actual}"
    print("  PASSED: path composition matches get_relation")


def test_empty_and_missing():
    """Edge cases: missing objects, same object."""
    m = SpatialMap()
    m.add_relation("A", "B", "above")

    assert m.get_reasoning_steps("X", "A") == ""
    assert m.get_reasoning_steps("A", "X") == ""
    print("  PASSED: missing objects return empty string")


def test_return_nodes_ordering():
    """_find_all_shortest_paths with return_nodes returns ordered node sequences."""
    m = SpatialMap()
    m.add_relation("B", "A", "above")
    m.add_relation("C", "B", "right")

    paths, node_seqs = m._find_all_shortest_paths("A", "C", return_nodes=True)
    assert len(paths) == 1
    assert len(node_seqs) == 1
    assert node_seqs[0] == ["A", "B", "C"]
    assert paths[0] == ["above", "right"]
    print("  PASSED: return_nodes gives ordered sequences")


if __name__ == "__main__":
    print("=" * 60)
    print("Reasoning Steps Tests")
    print("=" * 60)

    test_1_hop()
    test_2_hop()
    test_3_hop()
    test_4_hop()
    test_consistency()
    test_empty_and_missing()
    test_return_nodes_ordering()

    print()
    print("All tests passed!")
