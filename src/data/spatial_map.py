import io
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

# Use try/except to support both relative and absolute imports
try:
    from .spatial_constants import (
        BASE_DIRECTIONS,
        CARDINAL_SYNONYMS,
        CLOCK_SYNONYMS,
        DIRECTION_SYNONYMS,
        INVERSE_DIRECTIONS,
        LABEL_ALIASES,
        POSITION_OFFSETS,
    )
    from .spatial_utils import (
        extract_horizontal_component,
        extract_vertical_component,
        get_inverse_direction,
        infer_relation_from_graph,
        infer_relation_from_path,
        normalize_positions,
    )
except ImportError:
    from spatial_constants import (
        BASE_DIRECTIONS,
        CARDINAL_SYNONYMS,
        CLOCK_SYNONYMS,
        DIRECTION_SYNONYMS,
        INVERSE_DIRECTIONS,
        LABEL_ALIASES,
        POSITION_OFFSETS,
    )
    from spatial_utils import (
        extract_horizontal_component,
        extract_vertical_component,
        get_inverse_direction,
        infer_relation_from_graph,
        infer_relation_from_path,
        normalize_positions,
    )


class SpatialMap:
    # Import shared constants as class attributes for backward compatibility
    POSITION_OFFSETS = POSITION_OFFSETS
    INVERSE_DIRECTIONS = INVERSE_DIRECTIONS
    DIRECTION_SYNONYMS = DIRECTION_SYNONYMS
    CARDINAL_SYNONYMS = CARDINAL_SYNONYMS
    CLOCK_SYNONYMS = CLOCK_SYNONYMS
    LABEL_ALIASES = LABEL_ALIASES

    def __init__(self):
        # Core graph structure - adjacency list
        self.relations = defaultdict(list)

        # Stated relations for description generation (preserves order)
        self.stated_relations = []

        # O(1) edge lookup for validation: {(obj1, obj2): direction}
        self._edge_map = {}

        # Lazy-computed caches (invalidated on modification)
        self._positions_cache = None
        self._objects_cache = None
        self._positions_dirty = True

        # Metadata
        self.ambiguity_count = 0

    @property
    def objects(self) -> set:
        """Derive objects from relations on-demand (no redundant storage)"""
        if self._objects_cache is None:
            self._objects_cache = set()
            for obj in self.relations.keys():
                self._objects_cache.add(obj)
        return self._objects_cache

    @property
    def positions(self) -> Dict[str, Tuple[int, int]]:
        """Lazy position computation - only recompute when dirty"""
        if self._positions_dirty or self._positions_cache is None:
            self._update_positions()
            self._positions_dirty = False
        return self._positions_cache

    def _invalidate_caches(self):
        """Called when graph structure changes"""
        self._positions_dirty = True
        self._objects_cache = None

    def add_relation(self, obj1: str, obj2: str, direction: str):
        """Add a relation: obj1 is in the given direction relative to obj2"""
        if obj1 == obj2:
            raise ValueError(
                f"Cannot add relation: object '{obj1}' cannot have a spatial relation with itself"
            )

        normalized_dir = direction.lower().strip()

        if (
            normalized_dir == direction
            and normalized_dir not in self.DIRECTION_SYNONYMS
        ):
            valid_dirs = ", ".join(sorted(self.DIRECTION_SYNONYMS.keys()))
            raise ValueError(
                f"Invalid direction '{direction}'. Valid directions are: {valid_dirs}"
            )

        # Use O(1) edge lookup to check for duplicate
        edge_key = (obj1, obj2)
        if edge_key in self._edge_map:
            raise ValueError(
                f"Relation already exists: {obj1} is {self._edge_map[edge_key]} of {obj2}"
            )

        # Check for conflicting relation using O(1) lookup
        existing_rel = self._check_conflicting_relation(obj1, obj2, normalized_dir)
        if existing_rel:
            raise ValueError(
                f"Conflicting relation: cannot add '{obj1} is {normalized_dir} of {obj2}' "
                f"because '{obj1} is {existing_rel} of {obj2}' already exists"
            )

        # Add to adjacency list (bidirectional)
        inverse_dir = get_inverse_direction(normalized_dir)
        self.relations[obj1].append((obj2, inverse_dir))
        self.relations[obj2].append((obj1, normalized_dir))

        # Add to edge map for O(1) lookups
        self._edge_map[edge_key] = normalized_dir
        self._edge_map[(obj2, obj1)] = inverse_dir

        # Track stated relations for description generation
        self.stated_relations.append((obj1, obj2, normalized_dir))

        # Invalidate caches - positions will be computed lazily when needed
        self._invalidate_caches()

        # Check ambiguity (requires position computation)
        is_ambiguous_after = not self.has_unique_representation()
        if is_ambiguous_after:
            self.ambiguity_count += 1

        # Validate geometric consistency
        if not self._validate_geometric_consistency():
            # Rollback changes
            self.relations[obj1].pop()
            self.relations[obj2].pop()
            del self._edge_map[edge_key]
            del self._edge_map[(obj2, obj1)]
            self.stated_relations.pop()
            if is_ambiguous_after:
                self.ambiguity_count -= 1
            if not self.relations[obj1]:
                del self.relations[obj1]
            if not self.relations[obj2]:
                del self.relations[obj2]
            self._invalidate_caches()
            raise ValueError(
                f"Cannot add relation '{obj1} is {normalized_dir} of {obj2}': "
                f"it creates a geometrically inconsistent spatial arrangement"
            )

    def _validate_geometric_consistency(self) -> bool:
        """Validate that all stated relations match the computed positions."""
        if not self.positions:
            return True

        for obj1, obj2, direction in self.stated_relations:
            if obj1 not in self.positions or obj2 not in self.positions:
                continue

            pos1 = self.positions[obj1]
            pos2 = self.positions[obj2]

            expected_offset = self._get_offset(direction)
            actual_offset = (pos1[0] - pos2[0], pos1[1] - pos2[1])

            if not self._offset_matches_direction(
                actual_offset, expected_offset, direction
            ):
                return False

        return True

    def _offset_matches_direction(
        self, actual: Tuple[int, int], expected: Tuple[int, int], direction: str
    ) -> bool:
        """Check if actual offset matches the expected direction."""
        dx_actual, dy_actual = actual
        dx_expected, dy_expected = expected

        if direction in ["above", "below"]:
            return dx_actual == 0 and (
                dy_actual * dy_expected > 0 or dy_actual == dy_expected
            )
        elif direction in ["left", "right"]:
            return dy_actual == 0 and (
                dx_actual * dx_expected > 0 or dx_actual == dx_expected
            )
        else:
            if dx_expected != 0:
                if dx_actual == 0 or (dx_actual * dx_expected <= 0):
                    return False
            if dy_expected != 0:
                if dy_actual == 0 or (dy_actual * dy_expected <= 0):
                    return False
            return True

    def _check_conflicting_relation(
        self, obj1: str, obj2: str, new_direction: str
    ) -> Optional[str]:
        """Check if adding this relation would conflict with existing relations using O(1) lookup"""
        conflicts = {
            "above": ["below"],
            "below": ["above"],
            "left": ["right"],
            "right": ["left"],
            "upper-left": ["lower-right", "right", "below"],
            "upper-right": ["lower-left", "left", "below"],
            "lower-left": ["upper-right", "right", "above"],
            "lower-right": ["upper-left", "left", "above"],
        }

        # O(1) lookup in edge map
        edge_key = (obj1, obj2)
        if edge_key in self._edge_map:
            existing_dir = self._edge_map[edge_key]
            if existing_dir in conflicts.get(new_direction, []):
                return existing_dir

        return None

    def _generate_description(
        self, obj1: str, obj2: str, direction: str, mode: str = "spatial"
    ) -> Tuple[str, bool, bool]:
        """Generate natural language description with random synonyms.

        Args:
            obj1: First object
            obj2: Second object
            direction: Spatial direction
            mode: Description mode - "spatial", "clock", "cardinal", or combinations
                  "spatial" - use only DIRECTION_SYNONYMS (base directional terms)
                  "clock" - use only CLOCK_SYNONYMS (clock positions)
                  "cardinal" - use only CARDINAL_SYNONYMS (north/south/east/west)
                  "spatial+cardinal" - randomly choose from both spatial and cardinal
                  "spatial+clock" - randomly choose from both spatial and clock
                  "cardinal+clock" - randomly choose from both cardinal and clock
                  "spatial+cardinal+clock" - randomly choose from all three

        Returns:
            Tuple of (description, used_cardinal, used_clock):
                - description: Natural language description string in format "A is {RELATION} B"
                - used_cardinal: Boolean indicating if a cardinal term was used
                - used_clock: Boolean indicating if a clock term was used
        """
        # Cache synonym lookups (avoid repeated .get() calls)
        spatial_syns = self.DIRECTION_SYNONYMS.get(direction, [])
        cardinal_syns = self.CARDINAL_SYNONYMS.get(direction, [])
        clock_syns = self.CLOCK_SYNONYMS.get(direction, [])

        # Parse mode to determine available synonym types
        mode_parts = mode.split("+")

        # Build list of available synonym pools (not flattened - keep them separate)
        available_pools = []
        if "spatial" in mode_parts and spatial_syns:
            available_pools.append(("spatial", spatial_syns))
        if "cardinal" in mode_parts and cardinal_syns:
            available_pools.append(("cardinal", cardinal_syns))
        if "clock" in mode_parts and clock_syns:
            available_pools.append(("clock", clock_syns))

        # If no pools available, default to spatial
        if not available_pools:
            available_pools = [("spatial", spatial_syns)]

        # First randomly choose which TYPE to use (equal probability for each type)
        # Then randomly choose a phrase from that type
        chosen_type, chosen_pool = random.choice(available_pools)
        chosen_phrase = random.choice(chosen_pool) if chosen_pool else direction

        # Track which type was used
        used_cardinal = chosen_type == "cardinal"
        used_clock = chosen_type == "clock"

        description = f"{obj1} is {chosen_phrase} {obj2}"

        return description, used_cardinal, used_clock

    def _get_offset(self, direction: str) -> Tuple[int, int]:
        return self.POSITION_OFFSETS.get(direction, (0, 0))

    def _update_positions(self):
        """Compute and compact object positions using BFS placement."""
        if not self.objects:
            self._positions_cache = {}
            return

        if not self.relations:
            self._positions_cache = {obj: (i, 0) for i, obj in enumerate(self.objects)}
            return

        self._positions_cache = {}
        components = self._get_connected_components()

        # Position each component with offset
        x_offset = 0
        for component in components:
            self._position_component_simple(component, x_offset)
            if component and self._positions_cache:
                max_x = max(
                    self._positions_cache[obj][0]
                    for obj in component
                    if obj in self._positions_cache
                )
                x_offset = max_x + 2

        # Compact grid to remove empty rows/columns
        self._positions_cache = normalize_positions(self._positions_cache)

    def _position_component_simple(self, component: Set[str], x_offset: int):
        """Position objects in a component using BFS. Gaps removed later by compaction."""
        placed = set()

        # Start from most connected object
        start_obj = max(component, key=lambda x: len(self.relations.get(x, [])))
        self._positions_cache[start_obj] = (x_offset, 0)
        placed.add(start_obj)

        queue = deque([start_obj])

        while queue:
            current = queue.popleft()
            current_pos = self._positions_cache[current]

            for neighbor, direction in self.relations.get(current, []):
                if neighbor in component and neighbor not in placed:
                    offset = self._get_offset(direction)
                    new_pos = (current_pos[0] + offset[0], current_pos[1] + offset[1])

                    # Robust collision avoidance: keep shifting until unoccupied position found
                    max_shifts = 100  # Safety limit to prevent infinite loops
                    shifts = 0
                    while (
                        new_pos in self._positions_cache.values()
                        and shifts < max_shifts
                    ):
                        new_pos = (new_pos[0] + offset[0], new_pos[1] + offset[1])
                        shifts += 1

                    self._positions_cache[neighbor] = new_pos
                    placed.add(neighbor)
                    queue.append(neighbor)

    def _get_connected_components(self) -> List[Set[str]]:
        """Get all connected components in the spatial map"""
        if not self.objects:
            return []

        placed = set()
        components = []

        for obj in self.objects:
            if obj not in placed:
                component = set()
                queue = deque([obj])
                component.add(obj)
                placed.add(obj)

                while queue:
                    current = queue.popleft()
                    for neighbor, _ in self.relations.get(current, []):
                        if neighbor not in placed:
                            component.add(neighbor)
                            placed.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

        return components

    def render(self, style: str = "panel") -> str:
        """Render spatial map as ASCII art. Positions already compacted.

        Args:
            style: "simple", "grid", or "panel" (default)

        Returns:
            ASCII art string
        """
        if not self.positions:
            return "No objects to display"

        # Check for disconnected components and raise exception
        components = self._get_connected_components()

        if len(components) > 1:
            component_descriptions = []
            for i, component in enumerate(components, 1):
                objects_list = sorted(list(component))
                component_descriptions.append(
                    f"Component {i}: {', '.join(objects_list)}"
                )

            error_message = (
                f"Cannot render spatial map: Found {len(components)} disconnected components with no spatial relationships between them:\n"
                + "\n".join(f"  • {desc}" for desc in component_descriptions)
                + "\n\nPlease add spatial relationships to connect all components before rendering."
            )
            raise ValueError(error_message)

        # Positions are already compacted, just get dimensions
        max_x = max(pos[0] for pos in self.positions.values())
        max_y = max(pos[1] for pos in self.positions.values())

        if style == "simple":
            return self._render_simple(max_x, max_y)
        elif style == "grid":
            return self._render_grid(max_x, max_y)
        else:  # default to "panel"
            return self._render_box(max_x, max_y)

    def _render_simple(self, max_x: int, max_y: int) -> str:
        max_obj_len = max(len(str(obj)) for obj in self.objects)
        grid_width = (max_x + 1) * (max_obj_len + 2)
        grid_height = max_y + 1

        grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

        for obj, (x, y) in self.positions.items():
            grid_x = x * (max_obj_len + 2)
            grid_y = y

            if 0 <= grid_y < grid_height and 0 <= grid_x < grid_width:
                obj_str = str(obj)
                for i, char in enumerate(obj_str):
                    if grid_x + i < grid_width:
                        grid[grid_y][grid_x + i] = char

        result = []
        for row in grid:
            line = "".join(row).rstrip()
            if line:
                result.append(line)

        return "\n".join(result)

    def _render_grid(self, max_x: int, max_y: int) -> str:
        max_obj_len = max(len(str(obj)) for obj in self.objects)
        cell_width = max_obj_len + 2
        cell_height = 2

        grid_width = (max_x + 1) * cell_width + 1
        grid_height = (max_y + 1) * cell_height + 1

        grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

        # Draw horizontal lines
        for y in range(max_y + 2):
            for x in range(grid_width):
                grid[y * cell_height][x] = "-"

        # Draw vertical lines
        for x in range(max_x + 2):
            for y in range(grid_height):
                grid[y][x * cell_width] = "|"

        # Draw intersections
        for y in range(max_y + 2):
            for x in range(max_x + 2):
                grid[y * cell_height][x * cell_width] = "+"

        # Place objects centered in cells
        for obj, (x, y) in self.positions.items():
            obj_str = str(obj)
            start_x = x * cell_width + (cell_width - len(obj_str)) // 2
            start_y = y * cell_height + 1

            for i, char in enumerate(obj_str):
                if start_x + i < grid_width - 1:
                    grid[start_y][start_x + i] = char

        return "\n".join("".join(row) for row in grid)

    def _render_box(self, max_x: int, max_y: int) -> str:
        # Minimal spacing
        max_obj_len = max(len(str(obj)) for obj in self.objects)
        cell_width = max_obj_len + 2
        cell_height = 1

        grid_width = (max_x + 1) * cell_width + 3
        grid_height = (max_y + 1) * cell_height + 2

        grid = [[" " for _ in range(grid_width)] for _ in range(grid_height)]

        # Draw outer border
        for x in range(1, grid_width - 1):
            grid[0][x] = "─"
            grid[grid_height - 1][x] = "─"
        for y in range(1, grid_height - 1):
            grid[y][0] = "│"
            grid[y][grid_width - 1] = "│"

        # Draw corners
        grid[0][0] = "┌"
        grid[0][grid_width - 1] = "┐"
        grid[grid_height - 1][0] = "└"
        grid[grid_height - 1][grid_width - 1] = "┘"

        # Place objects without individual borders
        for obj, (x, y) in self.positions.items():
            obj_str = str(obj)
            text_x = x * cell_width + 2
            text_y = y * cell_height + 1

            for i, char in enumerate(obj_str):
                if text_x + i < grid_width - 1:
                    grid[text_y][text_x + i] = char

        return "\n".join("".join(row) for row in grid)

    def _are_objects_connected(self, obj1: str, obj2: str) -> bool:
        """Check if two objects are in the same connected component"""
        if obj1 not in self.objects or obj2 not in self.objects:
            return False

        components = self._get_connected_components()
        for component in components:
            if obj1 in component and obj2 in component:
                return True
        return False

    def get_relation(self, obj1: str, obj2: str) -> Optional[str]:
        """Get the spatial relationship between two objects.

        This method uses graph-based inference to maintain consistency with
        the stated relation structure. It traverses the relation graph using
        BFS to find the shortest path and infers the relationship through
        transitivity.

        Args:
            obj1: Source object
            obj2: Target object

        Returns:
            Spatial relation string (e.g., "above", "upper-left"), or None if
            objects don't exist

        Raises:
            ValueError: If objects are in disconnected components
        """
        if obj1 not in self.objects or obj2 not in self.objects:
            return None

        # Check if objects are in disconnected components
        if not self._are_objects_connected(obj1, obj2):
            raise ValueError(
                f"Cannot determine spatial relationship between '{obj1}' and '{obj2}': "
                f"they are in disconnected components with no spatial relationships between them. "
                f"Please add spatial relationships to connect their components."
            )

        # Use graph-based inference to maintain consistency with stated relations
        inferred_relation = infer_relation_from_graph(self.relations, obj1, obj2)

        return inferred_relation

    def get_vertical_relation(self, obj1: str, obj2: str) -> Optional[str]:
        """Get the vertical relationship between two objects (above, below, or same level).

        This method extracts the vertical component from the full spatial relation
        to ensure consistency with the overall spatial logic.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            "above" if obj1 is above obj2, "below" if obj1 is below obj2,
            "same level" if they're at the same vertical position, or None if objects don't exist
        """
        if obj1 not in self.objects or obj2 not in self.objects:
            return None

        # Check if objects are in disconnected components
        if not self._are_objects_connected(obj1, obj2):
            raise ValueError(
                f"Cannot determine vertical relationship between '{obj1}' and '{obj2}': "
                f"they are in disconnected components with no spatial relationships between them. "
                f"Please add spatial relationships to connect their components."
            )

        # Get the full relation and extract vertical component for consistency
        full_relation = self.get_relation(obj1, obj2)
        if full_relation is None or full_relation == "same position":
            return "same level"

        return extract_vertical_component(full_relation)

    def get_horizontal_relation(self, obj1: str, obj2: str) -> Optional[str]:
        """Get the horizontal relationship between two objects (left, right, or same column).

        This method extracts the horizontal component from the full spatial relation
        to ensure consistency with the overall spatial logic.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            "left" if obj1 is to the left of obj2, "right" if obj1 is to the right of obj2,
            "same column" if they're at the same horizontal position, or None if objects don't exist
        """
        if obj1 not in self.objects or obj2 not in self.objects:
            return None

        # Check if objects are in disconnected components
        if not self._are_objects_connected(obj1, obj2):
            raise ValueError(
                f"Cannot determine horizontal relationship between '{obj1}' and '{obj2}': "
                f"they are in disconnected components with no spatial relationships between them. "
                f"Please add spatial relationships to connect their components."
            )

        # Get the full relation and extract horizontal component for consistency
        full_relation = self.get_relation(obj1, obj2)
        if full_relation is None or full_relation == "same position":
            return "same column"

        return extract_horizontal_component(full_relation)

    def get_all_relations(self) -> Dict[Tuple[str, str], str]:
        """Get all relations between every pair of components in the spatial map.

        Returns:
            A dictionary mapping (obj1, obj2) tuples to their spatial relation strings
        """
        all_relations = {}

        # Update positions to ensure we have current spatial information
        self._update_positions()

        # Get all unique pairs of objects
        objects_list = list(self.objects)
        for i in range(len(objects_list)):
            for j in range(i + 1, len(objects_list)):
                obj1 = objects_list[i]
                obj2 = objects_list[j]

                # Try to get relation in both directions
                try:
                    relation = self.get_relation(obj1, obj2)
                    if relation:
                        all_relations[(obj1, obj2)] = relation

                    # Also get the inverse relation
                    relation_inv = self.get_relation(obj2, obj1)
                    if relation_inv:
                        all_relations[(obj2, obj1)] = relation_inv
                except ValueError:
                    # Objects are in disconnected components
                    pass

        return all_relations

    def get_detailed_relations(self) -> str:
        """Get all relations between components as a period-separated string.

        Returns:
            A string with all relations separated by periods using consistent format from DIRECTION_SYNONYMS
        """
        all_relations = self.get_all_relations()
        relation_strings = []

        for (obj1, obj2), relation in sorted(all_relations.items()):
            # Use the first synonym from DIRECTION_SYNONYMS for consistent formatting
            synonyms = self.DIRECTION_SYNONYMS.get(relation, [])
            if synonyms:
                relation_phrase = synonyms[0]
            else:
                # Fallback for any unexpected relation
                relation_phrase = relation

            relation_strings.append(f"{obj1} is {relation_phrase} {obj2}")

        return ". ".join(relation_strings) + "." if relation_strings else ""

    def get_reasoning_steps(self, obj1: str, obj2: str) -> str:
        """Get chain-of-thought reasoning steps for how obj1 relates to obj2.

        Traces the shortest path from obj2 (reference) to obj1 (target) and
        produces a natural language reasoning chain. Each hop is stated as a fact,
        and intermediate compositions are shown for paths longer than 2 hops.
        The final composition is omitted — the model must derive it.

        Args:
            obj1: Target object (the object we're asking about)
            obj2: Reference object

        Returns:
            Reasoning chain string, e.g.:
            - 1-hop: "B is above A."
            - 2-hop: "B is above A. C is to the right of B."
            - 3-hop: "B is above A. C is to the right of B. Since B is above A
                      and C is to the right of B, C is to the upper-right of A.
                      D is below C."
        """
        if obj1 not in self.objects or obj2 not in self.objects:
            return ""

        # BFS from obj2 (reference) toward obj1 (target), matching the
        # convention of infer_relation_from_graph
        paths, node_sequences = self._find_all_shortest_paths(
            obj2, obj1, return_nodes=True
        )
        if not paths:
            return ""

        return self._format_reasoning_chain(paths[0], node_sequences[0])

    def _format_reasoning_chain(
        self, path_directions: List[str], path_nodes: List[str]
    ) -> str:
        """Format a BFS path as a chain-of-thought reasoning string.

        Args:
            path_directions: Direction at each hop (length n)
            path_nodes: Ordered nodes from reference to target (length n+1)

        Returns:
            Natural language reasoning chain
        """
        n = len(path_directions)
        if n == 0:
            return ""

        def phrase(direction: str) -> str:
            synonyms = DIRECTION_SYNONYMS.get(direction, [])
            return synonyms[0] if synonyms else direction

        parts = []
        for i in range(n):
            # State the hop fact
            parts.append(
                f"{path_nodes[i + 1]} is {phrase(path_directions[i])} {path_nodes[i]}"
            )

            # Show intermediate composition after 2+ facts, but not for the last hop
            if i >= 1 and i < n - 1:
                prev_cumulative = infer_relation_from_path(path_directions[:i])
                curr_cumulative = infer_relation_from_path(path_directions[: i + 1])
                if prev_cumulative is not None and curr_cumulative is not None:
                    parts.append(
                        f"Since {path_nodes[i]} is {phrase(prev_cumulative)} "
                        f"{path_nodes[0]} and {path_nodes[i + 1]} is "
                        f"{phrase(path_directions[i])} {path_nodes[i]}, "
                        f"{path_nodes[i + 1]} is {phrase(curr_cumulative)} "
                        f"{path_nodes[0]}"
                    )

        return ". ".join(parts) + "."

    def describe_relations(self, mode: str = "spatial") -> Tuple[str, bool, bool]:
        """Return all natural language descriptions of relationships with random synonymous terms.

        Args:
            mode: Description mode - "spatial", "clock", "cardinal", or combinations
                  "spatial" - use only DIRECTION_SYNONYMS (base directional terms)
                  "clock" - use only CLOCK_SYNONYMS (clock positions)
                  "cardinal" - use only CARDINAL_SYNONYMS (north/south/east/west)
                  "spatial+cardinal" - randomly choose from both spatial and cardinal
                  "spatial+clock" - randomly choose from both spatial and clock
                  "cardinal+clock" - randomly choose from both cardinal and clock
                  "spatial+cardinal+clock" - randomly choose from all three

        Returns:
            Tuple of (description, uses_cardinal, uses_clock):
                - description: Natural language description of all stated relations
                - uses_cardinal: Boolean indicating if any cardinal term was used
                - uses_clock: Boolean indicating if any clock term was used
        """
        if not self.stated_relations:
            return "No relationships defined", False, False

        # Generate descriptions with random synonymous terms
        descriptions = []
        any_cardinal = False
        any_clock = False

        for obj1, obj2, direction in self.stated_relations:
            desc, used_cardinal, used_clock = self._generate_description(
                obj1, obj2, direction, mode
            )
            descriptions.append(desc)
            any_cardinal = any_cardinal or used_cardinal
            any_clock = any_clock or used_clock

        return ". ".join(descriptions) + ".", any_cardinal, any_clock

    def generate_jpg(
        self,
        filename: Optional[str] = None,
        cell_width: Optional[int] = None,
        cell_height: Optional[int] = None,
        font_size: int = 36,
        show_arrows: bool = True,
        show_grid: bool = True,
        quality: int = 95,
        bg_color: str = "white",
        text_color: str = "black",
        grid_color: str = "#E0E0E0",
        arrow_color: str = "#808080",
    ) -> Optional[bytes]:
        """Generate JPG visualization of spatial layout. Positions already compacted.

        Args:
            filename: Save path (None returns bytes)
            cell_width/height: Cell dimensions in pixels (None for auto)
            font_size: Label font size (default 36)
            show_arrows/grid: Display options (default True)
            quality: JPG quality 1-100 (default 95)
            *_color: Color customization

        Returns:
            Bytes if filename is None, else None
        """
        if not self.positions:
            raise ValueError(
                "No objects to visualize. Add spatial relationships first."
            )

        # Check for disconnected components
        components = self._get_connected_components()
        if len(components) > 1:
            raise ValueError(
                f"Cannot render: Found {len(components)} disconnected components"
            )

        # Get dimensions from compacted positions
        max_x = max(pos[0] for pos in self.positions.values())
        max_y = max(pos[1] for pos in self.positions.values())

        # Auto-calculate cell dimensions to match ASCII proportions
        if cell_width is None or cell_height is None:
            max_obj_len = max(len(str(obj)) for obj in self.objects)
            if cell_width is None:
                cell_width = (max_obj_len + 2) * 15
            if cell_height is None:
                cell_height = int(font_size * 2.5)

        # Calculate image dimensions with padding
        padding = cell_height // 3
        img_width = (max_x + 1) * cell_width + 2 * padding
        img_height = (max_y + 1) * cell_height + 2 * padding

        # Adaptive scaling for quality: 1.5x for small images, 2x for large
        total_size = max(img_width, img_height)
        scale_factor = 1.5 if total_size < 500 else 2
        img_width_scaled = int(img_width * scale_factor)
        img_height_scaled = int(img_height * scale_factor)
        img = Image.new("RGB", (img_width_scaled, img_height_scaled), bg_color)
        draw = ImageDraw.Draw(img, "RGBA")

        # Load font with fallback to default
        scaled_font_size = int(font_size * scale_factor)
        font_paths = [
            "Arial.ttf",
            "arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Avenir.ttc",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        font = None
        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, scaled_font_size)
                break
            except:
                continue

        if font is None:
            try:
                font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()

        # Draw grid lines
        if show_grid:
            for i in range(max_x + 2):
                x = int((padding + i * cell_width) * scale_factor)
                draw.line(
                    [
                        (x, int(padding * scale_factor)),
                        (x, int((img_height - padding) * scale_factor)),
                    ],
                    fill=grid_color,
                    width=int(scale_factor),
                )
            for i in range(max_y + 2):
                y = int((padding + i * cell_height) * scale_factor)
                draw.line(
                    [
                        (int(padding * scale_factor), y),
                        (int((img_width - padding) * scale_factor), y),
                    ],
                    fill=grid_color,
                    width=int(scale_factor),
                )

        # Draw relationship arrows
        if show_arrows:
            for obj1, obj2, direction in self.stated_relations:
                if obj1 in self.positions and obj2 in self.positions:
                    pos1 = self.positions[obj1]
                    pos2 = self.positions[obj2]

                    # Calculate arrow start and end points (scaled)
                    x1 = int(
                        (padding + pos1[0] * cell_width + cell_width // 2)
                        * scale_factor
                    )
                    y1 = int(
                        (padding + pos1[1] * cell_height + cell_height // 2)
                        * scale_factor
                    )
                    x2 = int(
                        (padding + pos2[0] * cell_width + cell_width // 2)
                        * scale_factor
                    )
                    y2 = int(
                        (padding + pos2[1] * cell_height + cell_height // 2)
                        * scale_factor
                    )

                    # Draw arrow line with antialiasing
                    draw.line(
                        [(x1, y1), (x2, y2)],
                        fill=arrow_color,
                        width=int(3 * scale_factor),
                    )

                    # Draw arrowhead (simple triangle)
                    import math

                    angle = math.atan2(y2 - y1, x2 - x1)
                    arrow_length = int(10 * scale_factor)
                    arrow_angle = 0.5

                    # Calculate arrowhead points
                    x3 = x2 - arrow_length * math.cos(angle - arrow_angle)
                    y3 = y2 - arrow_length * math.sin(angle - arrow_angle)
                    x4 = x2 - arrow_length * math.cos(angle + arrow_angle)
                    y4 = y2 - arrow_length * math.sin(angle + arrow_angle)

                    # Draw arrowhead
                    draw.polygon([(x2, y2), (x3, y3), (x4, y4)], fill=arrow_color)

        # Draw objects
        for obj, (x, y) in self.positions.items():
            # Calculate text position (centered in cell, scaled)
            text_x = int((padding + x * cell_width + cell_width // 2) * scale_factor)
            text_y = int((padding + y * cell_height + cell_height // 2) * scale_factor)

            # Get text bounding box for centering
            obj_str = str(obj)
            bbox = draw.textbbox((0, 0), obj_str, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw white background rectangle for text (for better readability)
            # Make the box fill most of the cell (about 92% of cell dimensions)
            box_width = int((cell_width * 0.92 * scale_factor) / 2)
            box_height = int((cell_height * 0.92 * scale_factor) / 2)
            # Draw rounded rectangle for better appearance
            bbox_rect = [
                text_x - box_width,
                text_y - box_height,
                text_x + box_width,
                text_y + box_height,
            ]
            draw.rounded_rectangle(
                bbox_rect,
                radius=int(3 * scale_factor),
                fill=bg_color,
                outline=text_color,
                width=int(2 * scale_factor),
            )

            # Draw the text (centered using anchor)
            draw.text(
                (text_x, text_y),
                obj_str,
                fill=text_color,
                font=font,
                anchor="mm",  # Middle-Middle anchor for perfect centering
            )

        # Downscale the image for smoother appearance (antialiasing)
        img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)

        # Save or return the image
        if filename:
            # Save to file with optimized settings
            # subsampling=1 uses 4:2:2 which reduces file size with minimal quality loss
            # progressive=False reduces file size
            img.save(
                filename,
                "JPEG",
                quality=quality,
                optimize=True,
                progressive=False,
                subsampling=1,
            )
            return None
        else:
            # Return as bytes
            img_bytes = io.BytesIO()
            img.save(
                img_bytes,
                "JPEG",
                quality=quality,
                optimize=True,
                progressive=False,
                subsampling=1,
            )
            return img_bytes.getvalue()

    @classmethod
    def get_all_spatial_relationships(cls) -> List[str]:
        """Return all valid spatial direction keys from DIRECTION_SYNONYMS"""
        return list(cls.DIRECTION_SYNONYMS.keys())

    @classmethod
    def get_label_aliases(cls, query_type: str = "full") -> Dict[str, List[str]]:
        """Return label aliases for evaluation/matching based on query type.

        Args:
            query_type: Type of query - "full", "vertical", or "horizontal"
                       "full" returns aliases for all 8 spatial directions
                       "vertical" returns aliases for above/below/same level
                       "horizontal" returns aliases for left/right/same column

        Returns:
            A dictionary mapping labels to their possible aliases
        """
        if query_type == "vertical":
            return {
                "above": cls.LABEL_ALIASES["above"],
                "below": cls.LABEL_ALIASES["below"],
                "same level": cls.LABEL_ALIASES["same level"],
            }
        elif query_type == "horizontal":
            return {
                "left": cls.LABEL_ALIASES["left"],
                "right": cls.LABEL_ALIASES["right"],
                "same column": cls.LABEL_ALIASES["same column"],
            }
        else:  # full spatial
            # Return only the 8 directional labels (exclude same level/same column)
            return {
                k: v
                for k, v in cls.LABEL_ALIASES.items()
                if k not in ["same level", "same column"]
            }

    def has_unique_representation(self) -> bool:
        """Check if the spatial layout has a unique representation.

        A layout has a unique representation when all pairwise spatial relations
        between objects can be uniquely determined from the stated relations.

        A layout is ambiguous when multiple objects have the same relation to a
        common reference object, but their mutual relation is not defined or inferable.

        Example of ambiguous layout:
            - "B is upper-right of A"
            - "C is upper-right of A"
            - No relation between B and C
            → B and C could be in various relative positions

        Returns:
            True if all relations are uniquely determinable, False otherwise
        """
        if len(self.objects) <= 1:
            return True

        objects_list = list(self.objects)

        # Check all pairs of objects
        for i in range(len(objects_list)):
            for j in range(i + 1, len(objects_list)):
                obj1, obj2 = objects_list[i], objects_list[j]

                # If any pair's relation is not uniquely inferable,
                # the entire layout is ambiguous
                if not self.is_relation_uniquely_inferable(obj1, obj2):
                    # print(f"Ambiguous relation between '{obj1}' and '{obj2}'")
                    return False

        return True

    def _check_path_agreement_component(
        self, all_paths: List[List[str]], component_extractor
    ) -> Optional[str]:
        """Check if all paths infer the same component (vertical or horizontal).

        IMPORTANT: We cannot use infer_relation_from_path here because it assumes
        unit magnitudes. For example, ['above', 'below'] would cancel to 'same level',
        but we don't know the actual magnitudes - it could be 5 steps above and 2 steps below,
        resulting in net 3 steps above.

        Instead, we accumulate component directions along each path and check if all paths
        have consistent net direction (or all are same level/same column).

        Args:
            all_paths: List of paths, where each path is a list of relation strings
            component_extractor: Function to extract component (vertical/horizontal) from relation

        Returns:
            The agreed-upon component, or None if paths disagree
        """
        inferred_components = set()

        for path_relations in all_paths:
            # Extract components from each step in the path
            components_in_path = [component_extractor(rel) for rel in path_relations]

            # Count directional components (ignore "same level" / "same column")
            component_counts = {}
            for comp in components_in_path:
                if comp not in ["same level", "same column"]:
                    component_counts[comp] = component_counts.get(comp, 0) + 1

            # Determine net component for this path
            # If we have both opposing directions (e.g., "above" and "below"),
            # we cannot determine the net direction without knowing magnitudes
            if component_extractor == extract_vertical_component:
                if "above" in component_counts and "below" in component_counts:
                    # Ambiguous: don't know which dominates
                    return None
                elif "above" in component_counts:
                    inferred_components.add("above")
                elif "below" in component_counts:
                    inferred_components.add("below")
                else:
                    # All steps are "same level"
                    inferred_components.add("same level")
            else:  # horizontal
                if "left" in component_counts and "right" in component_counts:
                    # Ambiguous: don't know which dominates
                    return None
                elif "left" in component_counts:
                    inferred_components.add("left")
                elif "right" in component_counts:
                    inferred_components.add("right")
                else:
                    # All steps are "same column"
                    inferred_components.add("same column")

        # All paths must agree on exactly one component
        return inferred_components.pop() if len(inferred_components) == 1 else None

    def _find_siblings_with_same_component(
        self, obj: str, reference: str, component: str, component_extractor
    ) -> Set[str]:
        """Find objects with same vertical/horizontal component relation to reference.

        Example: If querying vertical and both B and C are "above" A (or "upper-left" A),
        returns {B, C} since they share the vertical component.

        Args:
            obj: The object we're comparing against
            reference: The common reference object
            component: The component to match (e.g., "above", "left")
            component_extractor: Function to extract component from full relation

        Returns:
            Set of sibling objects
        """
        siblings = set()
        for other_obj in self.objects:
            if other_obj == obj:
                continue
            if (other_obj, reference) in self._edge_map:
                other_relation = self._edge_map[(other_obj, reference)]
                other_component = component_extractor(other_relation)
                if other_component == component:
                    siblings.add(other_obj)
        return siblings

    def _has_independent_path(self, obj1: str, obj2: str, excluded_node: str) -> bool:
        """Check if path exists from obj1 to obj2 that doesn't go through excluded_node.

        This is more efficient than reconstructing paths - uses BFS with node exclusion.

        Args:
            obj1: Starting object
            obj2: Target object
            excluded_node: Node to exclude from paths

        Returns:
            True if an independent path exists, False otherwise
        """
        if obj1 == obj2:
            return True

        queue = deque([obj1])
        visited = {obj1, excluded_node}  # Exclude the node from the start!

        while queue:
            current = queue.popleft()

            for neighbor, _ in self.relations.get(current, []):
                if neighbor == obj2:
                    return True  # Found path without going through excluded_node
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def _check_extended_cluster_ambiguity_component(
        self, obj1: str, obj2: str, inferred_component: str, component_extractor
    ) -> bool:
        """Check for extended cluster ambiguity for vertical/horizontal components.

        Similar to _check_extended_cluster_ambiguity but checks component-level relations.

        Args:
            obj1: First object
            obj2: Reference object
            inferred_component: The inferred component (e.g., "above", "left")
            component_extractor: Function to extract component from full relation

        Returns:
            True if ambiguity detected, False otherwise
        """
        siblings = self._find_siblings_with_same_component(
            obj1, obj2, inferred_component, component_extractor
        )

        for sibling in siblings:
            if not self._are_objects_connected(obj1, sibling):
                return True

            if (obj1, sibling) not in self._edge_map and (
                sibling,
                obj1,
            ) not in self._edge_map:
                if not self._has_independent_path(obj1, sibling, excluded_node=obj2):
                    return True

        return False

    def is_relation_uniquely_inferable(self, obj1: str, obj2: str) -> bool:
        """Check if the relation between obj1 and obj2 is uniquely inferable.

        A full spatial relation is uniquely inferable if and only if BOTH
        the vertical and horizontal components are uniquely inferable.

        This compositional approach simplifies the logic:
        - A relation like "upper-left" is unique iff "above" AND "left" are both unique
        - Pure vertical/horizontal relations (e.g., "above", "left") are unique
          if their respective component is unique and the other is "same" (no constraint)

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            True if the relation can be uniquely determined, False otherwise
        """
        # Use compositional logic: relation is unique iff both components are unique
        return self.is_vertical_relation_uniquely_inferable(
            obj1, obj2
        ) and self.is_horizontal_relation_uniquely_inferable(obj1, obj2)

    def is_vertical_relation_uniquely_inferable(self, obj1: str, obj2: str) -> bool:
        """Check if the vertical relation between obj1 and obj2 is uniquely inferable.

        A vertical relation is uniquely inferable if all shortest paths between
        the objects infer the same vertical component (above/below/same level).

        This is useful for checking whether the vertical aspect of a spatial
        relationship can be uniquely determined, even if the full relation
        (including horizontal component) might be ambiguous.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            True if the vertical relation can be uniquely determined, False otherwise
        """
        # Step 1: Basic validation
        if obj1 not in self.objects or obj2 not in self.objects:
            return False
        if obj1 == obj2:
            return True

        # Step 2: Direct relation is always unique
        if (obj1, obj2) in self._edge_map:
            return True

        # Step 3: Objects must be connected
        if not self._are_objects_connected(obj1, obj2):
            return False

        # Step 4: Find all shortest paths and check vertical component agreement
        all_paths = self._find_all_shortest_paths(obj2, obj1)
        if not all_paths:
            return False

        inferred_vertical = self._check_path_agreement_component(
            all_paths, extract_vertical_component
        )
        if inferred_vertical is None:
            return False  # Paths disagree on vertical component

        # Step 5: Check for extended cluster ambiguity at component level
        if self._check_extended_cluster_ambiguity_component(
            obj1, obj2, inferred_vertical, extract_vertical_component
        ):
            return False

        return True

    def is_horizontal_relation_uniquely_inferable(self, obj1: str, obj2: str) -> bool:
        """Check if the horizontal relation between obj1 and obj2 is uniquely inferable.

        A horizontal relation is uniquely inferable if all shortest paths between
        the objects infer the same horizontal component (left/right/same column).

        This is useful for checking whether the horizontal aspect of a spatial
        relationship can be uniquely determined, even if the full relation
        (including vertical component) might be ambiguous.

        Args:
            obj1: First object
            obj2: Second object

        Returns:
            True if the horizontal relation can be uniquely determined, False otherwise
        """
        # Step 1: Basic validation
        if obj1 not in self.objects or obj2 not in self.objects:
            return False
        if obj1 == obj2:
            return True

        # Step 2: Direct relation is always unique
        if (obj1, obj2) in self._edge_map:
            return True

        # Step 3: Objects must be connected
        if not self._are_objects_connected(obj1, obj2):
            return False

        # Step 4: Find all shortest paths and check horizontal component agreement
        all_paths = self._find_all_shortest_paths(obj2, obj1)
        if not all_paths:
            return False

        inferred_horizontal = self._check_path_agreement_component(
            all_paths, extract_horizontal_component
        )
        if inferred_horizontal is None:
            return False  # Paths disagree on horizontal component

        # Step 5: Check for extended cluster ambiguity at component level
        if self._check_extended_cluster_ambiguity_component(
            obj1, obj2, inferred_horizontal, extract_horizontal_component
        ):
            return False

        return True

    def _find_all_shortest_paths(
        self, start: str, target: str, return_nodes: bool = False
    ):
        """Find all shortest paths from start to target using BFS.

        This is used to detect cluster ambiguity: when multiple shortest paths
        exist between two objects that would infer different relations.

        Args:
            start: Starting object
            target: Target object
            return_nodes: If True, also returns ordered node sequences for each path.

        Returns:
            If return_nodes is False: List of paths, where each path is a list of relation strings
            If return_nodes is True: Tuple of (paths, node_sequences) where node_sequences[i]
                                     is the ordered list of nodes for paths[i]
                                     (e.g., [start, intermediate..., target])
        """
        if start not in self.objects or target not in self.objects:
            return ([], []) if return_nodes else []

        if start == target:
            return ([], [[start]]) if return_nodes else [[]]

        # BFS to find shortest distance first
        # Each queue item is (current_node, path_relations, path_nodes)
        queue = deque([(start, [], [start])])
        visited_at_distance = {start: 0}
        shortest_distance = None
        all_paths = []
        all_path_nodes = [] if return_nodes else None

        while queue:
            current, path, nodes = queue.popleft()
            current_distance = len(path)

            # If we've found paths and this path is longer, stop
            if shortest_distance is not None and current_distance > shortest_distance:
                break

            # Explore neighbors
            for neighbor, direction in self.relations.get(current, []):
                new_path = path + [direction]
                new_nodes = nodes + [neighbor] if return_nodes else None
                new_distance = len(new_path)

                # Found the target
                if neighbor == target:
                    if shortest_distance is None:
                        shortest_distance = new_distance
                    if new_distance == shortest_distance:
                        all_paths.append(new_path)
                        if return_nodes:
                            all_path_nodes.append(new_nodes)
                    continue

                # Only visit if this is the first time or we're at the same distance
                if (
                    neighbor not in visited_at_distance
                    or visited_at_distance[neighbor] == new_distance
                ):
                    if neighbor not in visited_at_distance:
                        visited_at_distance[neighbor] = new_distance
                    queue.append((neighbor, new_path, new_nodes))

        if return_nodes:
            return all_paths, all_path_nodes
        else:
            return all_paths


if __name__ == "__main__":
    grid1 = SpatialMap()
    grid1.add_relation("A", "B", "left")
    grid1.add_relation("C", "A", "above")
    print(grid1.render("simple"))
    print()

    grid2 = SpatialMap()
    grid2.add_relation("X", "Y", "right")
    grid2.add_relation("Z", "Y", "below")
    grid2.add_relation("W", "X", "upper-left")
    print(grid2.render("grid"))
    print()

    grid3 = SpatialMap()
    grid3.add_relation("P", "Q", "upper-left")
    grid3.add_relation("R", "Q", "right")
    grid3.add_relation("S", "R", "below")
    print(grid3.render("panel"))
    print()

    # Test the new __str__ method
    print("\nTesting __str__ method:")
    print(grid3)
    print()

    # Using get_relation function
    print("Relationship examples:")
    print(f"P is {grid3.get_relation('P', 'Q')} of Q")
    print(f"Q is {grid3.get_relation('Q', 'P')} of P")
    print(f"R is {grid3.get_relation('R', 'S')} of S")
    print(f"S is {grid3.get_relation('S', 'R')} of R")
    print(f"P is {grid3.get_relation('P', 'R')} of R")
    print(f"P is {grid3.get_relation('P', 'S')} of S")

    # Using get_all_relations function
    print("\nAll relations using get_all_relations():")
    all_relations = grid3.get_all_relations()
    for (obj1, obj2), relation in sorted(all_relations.items()):
        print(f"{obj1} is {relation} of {obj2}")

    # Using get_detailed_relations function
    print("\nDetailed relations using get_detailed_relations():")
    detailed_relations = grid3.get_detailed_relations()
    print(detailed_relations)

    # Using describe_relations function
    print("\nRelationship descriptions:")
    print("Normal:", grid1.describe_relations())
    print("With clock terms:", grid3.describe_relations(mode="clock"))

    # Test JPG generation with grid3
    print("\n\nTesting JPG Generation:")
    print("-" * 40)

    try:
        # Generate with maximum quality
        print("\nGenerating spatial_grid_max_quality.jpg with maximum quality...")
        grid3.generate_jpg(
            "spatial_grid_max_quality.jpg",
            quality=100,
            show_grid=True,
            show_arrows=False,
        )
        print("✓ Generated spatial_grid_max_quality.jpg")

        # Generate as bytes
        jpg_bytes = grid3.generate_jpg()
        print(f"\n✓ Generated JPG as bytes ({len(jpg_bytes):,} bytes)")

    except Exception as e:
        print(f"✗ JPG generation failed: {e}")

    # Test validation features
    print("\n\nTesting validation features:")
    print("-" * 40)

    # Test 1: Self-relation
    test_grid = SpatialMap()
    try:
        test_grid.add_relation("A", "A", "above")
        print("ERROR: Self-relation should have been caught!")
    except ValueError as e:
        print(f"✓ Self-relation caught: {e}")

    # Test 2: Duplicate relation
    test_grid = SpatialMap()
    test_grid.add_relation("A", "B", "above")
    try:
        test_grid.add_relation("A", "B", "above")
        print("ERROR: Duplicate relation should have been caught!")
    except ValueError as e:
        print(f"✓ Duplicate relation caught: {e}")

    # Test 3: Conflicting relation
    test_grid = SpatialMap()
    test_grid.add_relation("A", "B", "above")
    try:
        test_grid.add_relation("A", "B", "below")
        print("ERROR: Conflicting relation should have been caught!")
    except ValueError as e:
        print(f"✓ Conflicting relation caught: {e}")

    # Test 4: Invalid direction
    test_grid = SpatialMap()
    try:
        test_grid.add_relation("A", "B", "somewhere")
        print("ERROR: Invalid direction should have been caught!")
    except ValueError as e:
        print(f"✓ Invalid direction caught: {e}")

    # Test 5: Valid variations of directions (with connected components)
    test_grid = SpatialMap()
    try:
        test_grid.add_relation("A", "B", "upper-left")
        test_grid.add_relation("C", "B", "above")
        test_grid.add_relation("E", "C", "upper-right")
        print("✓ All valid direction variations accepted")
        print(test_grid.render("simple"))
    except ValueError as e:
        print(f"ERROR: Valid directions rejected: {e}")

    # Test 6: Disconnected components validation
    print("\n--- Test 6: Disconnected Components ---")
    test_grid_disconnected = SpatialMap()
    test_grid_disconnected.add_relation("A", "B", "left")
    test_grid_disconnected.add_relation("X", "Y", "above")  # Disconnected from A-B

    print("Testing render with disconnected components:")
    try:
        test_grid_disconnected.render()
        print("ERROR: Should have raised exception for disconnected components!")
    except ValueError as e:
        print(f"✓ Disconnected components caught: {e}")

    print("\nTesting get_relation with disconnected components:")
    try:
        test_grid_disconnected.get_relation("A", "X")
        print("ERROR: Should have raised exception for disconnected components!")
    except ValueError as e:
        print(f"✓ get_relation for disconnected components caught: {e}")

    # print all spatial relationships
    spatial_data = SpatialMap.get_all_spatial_relationships()
    print("\nAll spatial relationships:", spatial_data)
