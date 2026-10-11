"""
Shared utility functions for spatial relationship processing.

This module contains common functions for position normalization and
spatial relation calculations used across the codebase.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Use try/except to support both relative and absolute imports
try:
    from .spatial_constants import INVERSE_DIRECTIONS
except ImportError:
    from spatial_constants import INVERSE_DIRECTIONS


def normalize_positions(
    positions: Dict[str, Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    """Compact positions by removing empty rows/columns and shifting to origin.

    Args:
        positions: Object positions (x, y)

    Returns:
        Compacted positions with consecutive coordinates from (0, 0)
    """
    if not positions:
        return {}

    # Get all occupied coordinates
    occupied_x = sorted(set(x for x, y in positions.values()))
    occupied_y = sorted(set(y for x, y in positions.values()))

    # Map sparse coordinates to consecutive indices
    x_compact = {sparse: dense for dense, sparse in enumerate(occupied_x)}
    y_compact = {sparse: dense for dense, sparse in enumerate(occupied_y)}

    return {obj: (x_compact[x], y_compact[y]) for obj, (x, y) in positions.items()}


def get_inverse_direction(direction: str) -> str:
    """Get the inverse of a spatial direction.

    Args:
        direction: Spatial direction (e.g., "above", "left")

    Returns:
        Inverse direction (e.g., "below", "right")

    Examples:
        >>> get_inverse_direction("above")
        'below'
        >>> get_inverse_direction("upper-left")
        'lower-right'
    """
    return INVERSE_DIRECTIONS.get(direction, direction)


def get_relation_from_positions(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
    """Determine spatial relationship: where pos1 is relative to pos2.

    Coordinate system: Y increases DOWNWARD (standard ASCII/screen coordinates)
    - Smaller y values are "above"
    - Larger y values are "below"

    Args:
        pos1: Position of first object (x, y)
        pos2: Position of second object (x, y)

    Returns:
        Direction string: above, below, left, right, upper-left, upper-right,
        lower-left, lower-right, or "same"

    Examples:
        >>> get_relation_from_positions((5, 3), (5, 5))
        'above'
        >>> get_relation_from_positions((7, 4), (3, 4))
        'right'
        >>> get_relation_from_positions((8, 2), (5, 5))
        'upper-right'
    """
    dx = pos1[0] - pos2[0]  # Positive dx: pos1 is to the right of pos2
    dy = pos1[1] - pos2[1]  # Positive dy: pos1 is below pos2 (y increases down)

    # Cardinal directions (aligned on one axis)
    if dx == 0:
        # Vertically aligned: check y-axis
        return "below" if dy > 0 else "above"
    elif dy == 0:
        # Horizontally aligned: check x-axis
        return "right" if dx > 0 else "left"
    else:
        # Diagonal: combine vertical and horizontal components
        vertical = "lower" if dy > 0 else "upper"
        horizontal = "right" if dx > 0 else "left"
        return f"{vertical}-{horizontal}"


def extract_vertical_component(full_relation: str) -> str:
    """Extract vertical component from a full spatial relation.

    This provides consistency by deriving vertical relations from the full
    8-direction relation rather than computing separately from coordinates.

    Args:
        full_relation: Full spatial relation (e.g., "above", "upper-left", "right")

    Returns:
        "above", "below", or "same level"

    Examples:
        >>> extract_vertical_component("above")
        'above'
        >>> extract_vertical_component("upper-left")
        'above'
        >>> extract_vertical_component("left")
        'same level'
        >>> extract_vertical_component("lower-right")
        'below'
    """
    # Relations with upward component
    if full_relation in ["above", "upper-left", "upper-right"]:
        return "above"
    # Relations with downward component
    elif full_relation in ["below", "lower-left", "lower-right"]:
        return "below"
    # Horizontal relations (no vertical component)
    else:
        return "same level"


def extract_horizontal_component(full_relation: str) -> str:
    """Extract horizontal component from a full spatial relation.

    This provides consistency by deriving horizontal relations from the full
    8-direction relation rather than computing separately from coordinates.

    Args:
        full_relation: Full spatial relation (e.g., "left", "upper-left", "above")

    Returns:
        "left", "right", or "same column"

    Examples:
        >>> extract_horizontal_component("left")
        'left'
        >>> extract_horizontal_component("upper-left")
        'left'
        >>> extract_horizontal_component("above")
        'same column'
        >>> extract_horizontal_component("lower-right")
        'right'
    """
    # Relations with leftward component
    if full_relation in ["left", "upper-left", "lower-left"]:
        return "left"
    # Relations with rightward component
    elif full_relation in ["right", "upper-right", "lower-right"]:
        return "right"
    # Vertical relations (no horizontal component)
    else:
        return "same column"


def get_vertical_relation(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
    """Determine vertical relationship: where pos1 is relative to pos2 on y-axis.

    This method first computes the full spatial relation, then extracts the
    vertical component. This ensures consistency with the full relation logic.

    Args:
        pos1: Position of first object (x, y)
        pos2: Position of second object (x, y)

    Returns:
        "above", "below", or "same level"

    Examples:
        >>> get_vertical_relation((5, 2), (5, 5))
        'above'
        >>> get_vertical_relation((3, 7), (8, 7))
        'same level'
    """
    full_relation = get_relation_from_positions(pos1, pos2)
    return extract_vertical_component(full_relation)


def get_horizontal_relation(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> str:
    """Determine horizontal relationship: where pos1 is relative to pos2 on x-axis.

    This method first computes the full spatial relation, then extracts the
    horizontal component. This ensures consistency with the full relation logic.

    Args:
        pos1: Position of first object (x, y)
        pos2: Position of second object (x, y)

    Returns:
        "left", "right", or "same column"

    Examples:
        >>> get_horizontal_relation((2, 5), (7, 5))
        'left'
        >>> get_horizontal_relation((5, 3), (5, 8))
        'same column'
    """
    full_relation = get_relation_from_positions(pos1, pos2)
    return extract_horizontal_component(full_relation)


def infer_relation_from_path(path_relations: list) -> Optional[str]:
    """Infer the composite spatial relation by combining relations along a path.

    This uses the relation graph structure to infer relations through transitivity.
    For example, if A is "above" B and B is "above" C, then A is "above" C.

    Args:
        path_relations: List of relations along a path from obj1 to obj2
                       e.g., ["above", "right", "below"]

    Returns:
        Inferred relation string, or None if relations cannot be composed

    Examples:
        >>> infer_relation_from_path(["above", "above"])
        'above'
        >>> infer_relation_from_path(["right", "right"])
        'right'
        >>> infer_relation_from_path(["above", "right"])
        'upper-right'
    """
    if not path_relations:
        return None

    # Start with the first relation
    cumulative_dx = 0  # Horizontal offset
    cumulative_dy = 0  # Vertical offset

    # Define offsets for each base direction
    direction_offsets = {
        "above": (0, -1),
        "below": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
        "upper-left": (-1, -1),
        "upper-right": (1, -1),
        "lower-left": (-1, 1),
        "lower-right": (1, 1),
    }

    # Accumulate all offsets along the path
    for relation in path_relations:
        if relation not in direction_offsets:
            # Unknown relation, cannot compose
            return None
        dx, dy = direction_offsets[relation]
        cumulative_dx += dx
        cumulative_dy += dy

    # Convert cumulative offsets back to a relation
    if cumulative_dx == 0 and cumulative_dy == 0:
        # Offsets canceled out (e.g., ["below", "above"] or ["left", "right"])
        # This indicates contradictory/circular path - cannot reliably infer relation
        # Return None instead of "same position" since objects are NOT at same position,
        # they just have a path with canceling directions
        return None
    elif cumulative_dx == 0:
        return "below" if cumulative_dy > 0 else "above"
    elif cumulative_dy == 0:
        return "right" if cumulative_dx > 0 else "left"
    else:
        vertical = "lower" if cumulative_dy > 0 else "upper"
        horizontal = "right" if cumulative_dx > 0 else "left"
        return f"{vertical}-{horizontal}"


def count_connected_components(objects: set, relations_graph: Dict[str, list]) -> int:
    """Count connected components in a graph of objects.

    Used to detect missing relations in descriptions - if there are N components,
    then N-1 additional relations are needed to connect all objects.

    Args:
        objects: Set of all object names to consider
        relations_graph: Adjacency list {obj: [(neighbor, direction), ...]}

    Returns:
        Number of connected components among the given objects

    Examples:
        >>> # A-B connected, C isolated -> 2 components
        >>> graph = {'a': [('b', 'above')], 'b': [('a', 'below')]}
        >>> count_connected_components({'a', 'b', 'c'}, graph)
        2
    """
    if not objects:
        return 0

    visited = set()
    num_components = 0

    for obj in objects:
        if obj not in visited:
            # BFS to mark all reachable objects
            num_components += 1
            queue = [obj]
            visited.add(obj)

            while queue:
                current = queue.pop(0)
                for neighbor, _ in relations_graph.get(current, []):
                    if neighbor not in visited and neighbor in objects:
                        visited.add(neighbor)
                        queue.append(neighbor)

    return num_components


def infer_relation_from_graph(
    relations_graph: Dict[str, list], obj1: str, obj2: str
) -> Optional[str]:
    """Infer spatial relation between two objects using graph traversal (BFS).

    This method uses the relation graph structure to find the shortest path
    and infer the relationship through transitivity.

    Graph format: relations_graph[A] = [(B, 'above')] means "from A's perspective, B is above"
    Query: get_relation(B, A) asks "where is B relative to A?"
    Answer: We look from A's perspective to find B

    Args:
        relations_graph: Adjacency list {obj: [(neighbor, direction), ...]}
                        Format: relations_graph[ref] = [(target, direction)]
                        Meaning: from ref's perspective, target is in direction
        obj1: Target object (the object we're asking about)
        obj2: Reference object (the reference point)

    Returns:
        Inferred spatial relation (where obj1 is relative to obj2), or None if no path exists

    Examples:
        Given graph: {'A': [('B', 'above')], 'B': [('A', 'below')]}
        After add_relation('B', 'A', 'above') which means "B is above A"
        >>> infer_relation_from_graph(graph, 'B', 'A')
        'above'  # B is above (relative to) A
    """
    if obj1 not in relations_graph or obj2 not in relations_graph:
        return None

    # BFS starting from obj2 (reference point) to find obj1 (target)
    # We traverse from the reference's perspective
    from collections import deque

    queue = deque([(obj2, [])])  # (current_obj, path_of_relations)
    visited = {obj2}

    while queue:
        current, path = queue.popleft()

        # Check all neighbors from current's perspective
        for neighbor, direction in relations_graph.get(current, []):
            if neighbor == obj1:
                # Found target! Infer relation from path
                final_path = path + [direction]
                return infer_relation_from_path(final_path)

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [direction]))

    # No path found
    return None


class UnionFind:
    """Disjoint Set Union with path compression and union by rank.

    Efficient data structure for tracking connectivity between objects.
    Used to determine which object pairs can have their relations inferred
    from stated relations (objects in the same component are inferrable).

    Time complexity:
        - find: O(α(n)) amortized (nearly constant)
        - union: O(α(n)) amortized
        - get_component_sizes: O(n)

    Examples:
        >>> uf = UnionFind(['a', 'b', 'c', 'd'])
        >>> uf.union('a', 'b')
        >>> uf.union('b', 'c')
        >>> uf.connected('a', 'c')
        True
        >>> uf.connected('a', 'd')
        False
        >>> uf.get_component_sizes()
        {3: 1, 1: 1}  # One component of size 3, one of size 1
    """

    __slots__ = ("parent", "rank")

    def __init__(self, elements):
        """Initialize Union-Find with given elements.

        Args:
            elements: Iterable of elements to track
        """
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x: str) -> str:
        """Find root of element with path compression.

        Args:
            x: Element to find root for

        Returns:
            Root element of the component containing x
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge components containing x and y.

        Uses union by rank to keep tree balanced.

        Args:
            x: First element
            y: Second element
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def connected(self, x: str, y: str) -> bool:
        """Check if two elements are in the same component.

        Args:
            x: First element
            y: Second element

        Returns:
            True if x and y are connected (in same component)
        """
        return self.find(x) == self.find(y)

    def get_component_sizes(self) -> Dict[str, int]:
        """Get size of each connected component.

        Returns:
            Dict mapping root element to component size
        """
        sizes = defaultdict(int)
        for x in self.parent:
            sizes[self.find(x)] += 1
        return dict(sizes)

    def get_components(self) -> Dict[str, Set[str]]:
        """Get all elements in each connected component.

        Returns:
            Dict mapping root element to set of elements in that component
        """
        components = defaultdict(set)
        for x in self.parent:
            components[self.find(x)].add(x)
        return dict(components)


def compute_missing_relations(
    stated_relations: List[Tuple[str, str, str]],
    all_objects: Set[str],
    relations_graph: Dict[str, List[Tuple[str, str]]] = None,
) -> Dict:
    """Compute missing relations using Union-Find for efficiency.

    A relation between two objects is "inferrable" if they are connected
    through any chain of stated relations. This function identifies which
    relations are needed to connect all disconnected components.

    Time: O(N + R * α(N)) ≈ O(N + R) where N = objects, R = stated relations
    Space: O(N)

    Args:
        stated_relations: List of (obj1, obj2, direction) tuples from description
        all_objects: Set of all object names in the grid
        relations_graph: Optional graph {obj: [(neighbor, direction), ...]} for
                        looking up actual relations between objects

    Returns:
        Dict with keys:
        - total_pairs: Total number of unique object pairs
        - inferrable_count: Pairs whose relation can be inferred
        - missing_count: Pairs whose relation cannot be inferred
        - needed_relations: List of example relations to connect components
        - min_relations_needed: Minimum additional relations for completeness
        - num_components: Number of connected components
        - coverage: Fraction of pairs that are inferrable (0.0 to 1.0)

    Examples:
        >>> stated = [('a', 'b', 'above'), ('b', 'c', 'left')]
        >>> objects = {'a', 'b', 'c', 'd'}
        >>> result = compute_missing_relations(stated, objects)
        >>> result['min_relations_needed']
        1  # One relation needed to connect 'd' to the other component
    """
    objects_list = list(all_objects)
    n = len(objects_list)

    if n < 2:
        return {
            "total_pairs": 0,
            "inferrable_count": 0,
            "missing_count": 0,
            "needed_relations": [],
            "min_relations_needed": 0,
            "num_components": n,
            "coverage": 1.0,
        }

    # Build Union-Find from stated relations
    uf = UnionFind(all_objects)

    for obj1, obj2, _ in stated_relations:
        if obj1 in all_objects and obj2 in all_objects:
            uf.union(obj1, obj2)

    # Get component information
    components = uf.get_components()  # {root: {obj1, obj2, ...}}
    num_components = len(components)

    # Calculate inferrable pairs from component sizes
    inferrable = sum(len(objs) * (len(objs) - 1) // 2 for objs in components.values())

    # Total unique pairs
    total = n * (n - 1) // 2
    missing_count = total - inferrable

    # Minimum relations needed = num_components - 1 (to form spanning tree)
    min_needed = max(0, num_components - 1)

    # Find example relations that would connect the components
    needed_relations = []
    if min_needed > 0 and relations_graph:
        component_list = list(components.values())
        # Connect components sequentially (component 0 to 1, 1 to 2, etc.)
        for i in range(len(component_list) - 1):
            comp1, comp2 = component_list[i], component_list[i + 1]
            # Pick first object from each component as representatives
            obj1 = next(iter(comp1))
            obj2 = next(iter(comp2))
            # Look up the actual relation from the grid
            relation = None
            for neighbor, direction in relations_graph.get(obj1, []):
                if neighbor == obj2:
                    relation = direction
                    break
            needed_relations.append(
                {"obj1": obj1, "obj2": obj2, "relation": relation or "unknown"}
            )

    return {
        "total_pairs": total,
        "inferrable_count": inferrable,
        "missing_count": missing_count,
        "needed_relations": needed_relations,
        "min_relations_needed": min_needed,
        "num_components": num_components,
        "coverage": inferrable / total if total > 0 else 1.0,
    }
