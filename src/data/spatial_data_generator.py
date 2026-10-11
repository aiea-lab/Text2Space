#!/usr/bin/env python3
"""
Spatial Data Generation Pipeline

Generates random spatial layouts with connected components and saves them to JSONL format.
"""

import argparse
import json
import random
import string
from typing import Dict, List, Optional, Set, Tuple

# Use try/except to support both relative and absolute imports
try:
    from .spatial_constants import CARDINAL_SYNONYMS, CLOCK_SYNONYMS, DIRECTION_SYNONYMS
    from .spatial_map import SpatialMap
except ImportError:
    from spatial_constants import CARDINAL_SYNONYMS, CLOCK_SYNONYMS, DIRECTION_SYNONYMS
    from spatial_map import SpatialMap


class BalancingTracker:
    """Tracks and balances multiple dimensions in dataset generation.

    Uses adaptive, formula-based balancing instead of arbitrary constants.
    Rejection probability grows naturally with deviation from targets.
    """

    def __init__(
        self,
        targets: Dict[str, float],
        num_bins: Dict[str, int] = None,
        category_values: Dict[str, List] = None,
        category_targets: Dict[str, Dict] = None,
        stratify_unique_by_components: bool = False,
    ):
        """
        Initialize balancing tracker.

        Args:
            targets: Dict mapping dimension names to target ratios (0.0 to 1.0)
                    e.g., {'unique': 0.5, 'direct': 0.5}
            num_bins: Dict mapping categorical dimension names to number of categories
                     e.g., {'num_components': 7, 'query_type': 3}
            category_values: Dict mapping categorical dimension to list of possible values
                           e.g., {'terminology': ['spatial', 'cardinal', ...]}
            category_targets: Optional weighted targets for categorical dimensions
                            e.g., {'num_components': {2: 0.025, 3: 0.075, 4: 0.18, ...}}
                            If not provided, uses uniform distribution
            stratify_unique_by_components: If True, balance uniqueness separately per component count
        """
        self.targets = targets
        self.num_bins = num_bins or {}
        self.category_values = category_values or {}
        self.category_targets = category_targets or {}
        self.stratify_unique_by_components = stratify_unique_by_components

        # Binary dimension counters (unique/direct etc.)
        self.binary_counts = {dim: 0 for dim in targets}

        # Categorical dimension counters (components, query types, etc.)
        self.categorical_counts = {dim: {} for dim in num_bins}

        # Stratified tracking: uniqueness per component count
        # Structure: {num_components: {'unique': count, 'non_unique': count, 'total': count}}
        self.stratified_unique_counts = {} if stratify_unique_by_components else None

        self.total_instances = 0

    def _get_warmup_period(self) -> int:
        """Calculate adaptive warmup period based on expected dataset size.

        Returns minimum of 10 instances or 5% of current total.
        This naturally adapts to any dataset size.
        """
        return max(10, int(self.total_instances * 0.05))

    def update(self, **kwargs):
        """Update counters with new instance properties."""
        self.total_instances += 1

        for dim, value in kwargs.items():
            if dim in self.binary_counts:
                if value:  # Boolean True
                    self.binary_counts[dim] += 1
            elif dim in self.categorical_counts:
                self.categorical_counts[dim][value] = (
                    self.categorical_counts[dim].get(value, 0) + 1
                )

        # Update stratified tracking if enabled
        if (
            self.stratified_unique_counts is not None
            and "unique" in kwargs
            and "num_components" in kwargs
        ):
            num_comp = kwargs["num_components"]
            is_unique = kwargs["unique"]

            if num_comp not in self.stratified_unique_counts:
                self.stratified_unique_counts[num_comp] = {
                    "unique": 0,
                    "non_unique": 0,
                    "total": 0,
                }

            self.stratified_unique_counts[num_comp]["total"] += 1
            if is_unique:
                self.stratified_unique_counts[num_comp]["unique"] += 1
            else:
                self.stratified_unique_counts[num_comp]["non_unique"] += 1

    def should_accept(self, base_aggressiveness: float = 15.0, **kwargs) -> bool:
        """
        Decide whether to accept instance based on current balance.

        Uses adaptive, formula-based rejection with quadratic scaling:
        rejection probability grows quadratically with deviation for stronger
        correction when far from targets.

        Args:
            base_aggressiveness: Base rejection strength (default: 15.0)
                               Higher values = stricter balancing
            **kwargs: Instance properties to evaluate

        Returns:
            True if instance should be accepted
        """
        # Accept during warmup period (adaptive to dataset size)
        warmup_period = self._get_warmup_period()
        if self.total_instances < warmup_period:
            return True

        # Progressive aggressiveness: grows with dataset size
        # Reaches full strength at 20x warmup period
        progress = min(1.0, self.total_instances / (warmup_period * 20))
        aggressiveness = base_aggressiveness * (0.5 + 0.5 * progress)

        # Check binary dimensions (unique, direct, etc.)
        for dim, target in self.targets.items():
            if dim in kwargs:
                value = kwargs[dim]

                # Stratified balancing for uniqueness by component count
                if (
                    self.stratified_unique_counts is not None
                    and dim == "unique"
                    and "num_components" in kwargs
                ):

                    num_comp = kwargs["num_components"]

                    # Initialize stratum if not seen yet
                    if num_comp not in self.stratified_unique_counts:
                        self.stratified_unique_counts[num_comp] = {
                            "unique": 0,
                            "non_unique": 0,
                            "total": 0,
                        }

                    stratum = self.stratified_unique_counts[num_comp]

                    # Skip if insufficient data for this stratum (need at least 10 instances)
                    if stratum["total"] < 10:
                        continue

                    # Calculate current ratio for this component count stratum
                    current_ratio = stratum["unique"] / stratum["total"]

                    # Apply balancing logic to this stratum
                    if value and current_ratio > target:
                        deviation = current_ratio - target
                        scaling_factor = 1.0 + (deviation * 10)
                        reject_prob = min(
                            0.95, deviation * aggressiveness * scaling_factor
                        )
                        if random.random() < reject_prob:
                            return False
                    elif not value and current_ratio < target:
                        deviation = target - current_ratio
                        scaling_factor = 1.0 + (deviation * 10)
                        reject_prob = min(
                            0.95, deviation * aggressiveness * scaling_factor
                        )
                        if random.random() < reject_prob:
                            return False

                    # Stratified check passed, skip global check for this dimension
                    continue

                # Standard global balancing for other dimensions
                current_ratio = self.binary_counts[dim] / self.total_instances

                # Reject if adding this value would worsen imbalance
                if value and current_ratio > target:
                    deviation = current_ratio - target
                    # Quadratic scaling: stronger rejection when further from target
                    # Boost factor: 1.5x at 5% off, 2.5x at 10% off, 5.5x at 20% off
                    scaling_factor = 1.0 + (deviation * 10)
                    reject_prob = min(0.95, deviation * aggressiveness * scaling_factor)
                    if random.random() < reject_prob:
                        return False

                elif not value and current_ratio < target:
                    deviation = target - current_ratio
                    scaling_factor = 1.0 + (deviation * 10)
                    reject_prob = min(0.95, deviation * aggressiveness * scaling_factor)
                    if random.random() < reject_prob:
                        return False

        # Check categorical dimensions (components, query types, etc.)
        for dim, num_bins in self.num_bins.items():
            if dim in kwargs:
                value = kwargs[dim]

                # Use weighted targets if available, otherwise uniform distribution
                if dim in self.category_targets and value in self.category_targets[dim]:
                    target_ratio = self.category_targets[dim][value]
                else:
                    target_ratio = 1.0 / num_bins

                current_count = self.categorical_counts[dim].get(value, 0)
                current_ratio = current_count / self.total_instances

                # Reject if overrepresented beyond tolerance
                tolerance = 0.5  # Allow 50% over target before rejecting
                if current_ratio > target_ratio * (1 + tolerance):
                    deviation = current_ratio - target_ratio
                    # Categorical: lighter touch for multi-category balancing
                    reject_prob = min(0.80, deviation * aggressiveness * 0.5)
                    if random.random() < reject_prob:
                        return False

        return True

    def get_most_needed_category(self, dimension: str) -> Optional[str]:
        """
        Get the most underrepresented category for a categorical dimension.
        Uses weighted targets if available, comparing actual vs target ratios.

        Args:
            dimension: The categorical dimension name (e.g., 'terminology')

        Returns:
            The category value that is most underrepresented, or None if not tracked
        """
        if dimension not in self.categorical_counts:
            return None

        # During warmup, use random selection
        if self.total_instances < self._get_warmup_period():
            return None

        current_counts = self.categorical_counts[dimension]
        has_weighted_targets = dimension in self.category_targets

        # If we know all possible values, check them all (including 0-count ones)
        if dimension in self.category_values:
            all_values = self.category_values[dimension]
            max_deficit = float("-inf")
            most_needed = None

            for value in all_values:
                count = current_counts.get(value, 0)  # 0 if not seen yet
                actual_ratio = count / self.total_instances

                # Calculate target ratio (weighted or uniform)
                if has_weighted_targets and value in self.category_targets[dimension]:
                    target_ratio = self.category_targets[dimension][value]
                else:
                    target_ratio = 1.0 / self.num_bins.get(dimension, len(all_values))

                # Find category with largest deficit (target - actual)
                deficit = target_ratio - actual_ratio
                if deficit > max_deficit:
                    max_deficit = deficit
                    most_needed = value

            return most_needed

        # Fallback: only check categories that have been seen
        if not current_counts:
            return None

        max_deficit = float("-inf")
        most_needed = None

        for category, count in current_counts.items():
            actual_ratio = count / self.total_instances

            # Calculate target ratio
            if has_weighted_targets and category in self.category_targets[dimension]:
                target_ratio = self.category_targets[dimension][category]
            else:
                target_ratio = 1.0 / self.num_bins.get(dimension, len(current_counts))

            deficit = target_ratio - actual_ratio
            if deficit > max_deficit:
                max_deficit = deficit
                most_needed = category

        return most_needed

    def get_stats(self) -> Dict:
        """Get current balance statistics."""
        stats = {"total": self.total_instances}

        # Binary stats
        for dim in self.binary_counts:
            if self.total_instances > 0:
                stats[dim] = {
                    "count": self.binary_counts[dim],
                    "ratio": self.binary_counts[dim] / self.total_instances,
                    "target": self.targets.get(dim, 0.5),
                }

        # Categorical stats
        for dim in self.categorical_counts:
            stats[dim] = dict(self.categorical_counts[dim])

        # Stratified stats for uniqueness by component count
        if self.stratified_unique_counts is not None:
            stats["stratified_unique"] = {}
            for num_comp in sorted(self.stratified_unique_counts.keys()):
                stratum = self.stratified_unique_counts[num_comp]
                if stratum["total"] > 0:
                    stats["stratified_unique"][num_comp] = {
                        "unique": stratum["unique"],
                        "non_unique": stratum["non_unique"],
                        "total": stratum["total"],
                        "unique_ratio": stratum["unique"] / stratum["total"],
                        "target": self.targets.get("unique", 0.5),
                    }

        return stats


class SpatialDataGenerator:
    # All possible terminology combinations (7 total)
    ALL_TERMINOLOGIES = [
        "spatial",
        "cardinal",
        "clock",
        "spatial+cardinal",
        "spatial+clock",
        "cardinal+clock",
        "spatial+cardinal+clock",
    ]

    # Mapping from desired terminology outcome to description mode
    # The mode determines which terms are AVAILABLE; actual usage is probabilistic
    TERMINOLOGY_TO_MODE = {
        "spatial": "spatial",
        "cardinal": "cardinal",
        "clock": "clock",
        "spatial+cardinal": "spatial+cardinal",
        "spatial+clock": "spatial+clock",
        "cardinal+clock": "cardinal+clock",
        "spatial+cardinal+clock": "spatial+cardinal+clock",
    }

    def __init__(
        self,
        max_components: int,
        max_hops: int,
        min_components: int = 2,
        min_hops: int = 1,
        exclude_same_level_column: bool = False,
    ):
        """
        Initialize the generator with constraints.

        Args:
            max_components: Maximum number of spatial components
            max_hops: Maximum number of relational hops (add_relation calls)
            min_components: Minimum number of spatial components (default: 2)
            exclude_same_level_column: If True, exclude queries about "same level"/"same column" relations (default: False)
            min_hops: Minimum number of relational hops (default: 1)
        """
        self.max_components = max_components
        self.max_hops = max_hops
        self.min_components = min_components
        self.min_hops = min_hops
        self.exclude_same_level_column = exclude_same_level_column
        self.spatial_relations = SpatialMap.get_all_spatial_relationships()

        # Precompute non-spatial terms for efficiency in _has_spatial_terms
        self._non_spatial_terms = set()
        for phrases in CARDINAL_SYNONYMS.values():
            self._non_spatial_terms.update(p.lower() for p in phrases)
        for phrases in CLOCK_SYNONYMS.values():
            self._non_spatial_terms.update(p.lower() for p in phrases)

    def generate_component_names(self, num_components: int) -> List[str]:
        """Generate component names as uppercase letters."""
        if num_components > 26:
            names = []
            for i in range(num_components):
                if i < 26:
                    names.append(string.ascii_uppercase[i])
                else:
                    first = (i - 26) // 26
                    second = (i - 26) % 26
                    names.append(
                        string.ascii_uppercase[first] + string.ascii_uppercase[second]
                    )
            return names
        return list(string.ascii_uppercase[:num_components])

    def _has_spatial_terms(self, description: str) -> bool:
        """Check if description contains spatial (base directional) terms.

        Returns True if any base spatial terms appear that are not cardinal or clock terms.
        """
        desc_lower = description.lower()

        # Check if any spatial-only phrase appears
        for phrases in DIRECTION_SYNONYMS.values():
            for phrase in phrases:
                if (
                    phrase.lower() not in self._non_spatial_terms
                    and phrase.lower() in desc_lower
                ):
                    return True

        return False

    def generate_connected_structure(
        self,
        target_num_components: Optional[int] = None,
        target_num_relations: Optional[int] = None,
    ) -> Tuple[SpatialMap, List[str], int]:
        """Generate a randomly connected spatial structure.

        Args:
            target_num_components: If specified, use this number of components (for balancing)
            target_num_relations: If specified, aim for this number of relations (for balancing)

        Returns:
            spatial_map: The generated SpatialMap instance
            components: List of component names
            num_relations: Actual number of relations added
        """
        # Use target if provided, otherwise random
        num_components = (
            target_num_components
            if target_num_components is not None
            else random.randint(self.min_components, self.max_components)
        )
        components = self.generate_component_names(num_components)

        spatial_map = SpatialMap()

        connected = set()
        unconnected = set(components)

        start = random.choice(components)
        connected.add(start)
        unconnected.remove(start)

        num_relations = 0
        # Use target if provided, otherwise use configured max
        target_hops = (
            target_num_relations if target_num_relations is not None else self.max_hops
        )
        max_attempts = target_hops * 3  # Allow 3x attempts to reach target
        attempts = 0

        while unconnected and num_relations < target_hops and attempts < max_attempts:
            attempts += 1

            if unconnected and connected:
                from_component = random.choice(list(connected))
                to_component = random.choice(list(unconnected))
                # Uniform random selection - let variety emerge naturally
                direction = random.choice(self.spatial_relations)

                try:
                    spatial_map.add_relation(to_component, from_component, direction)
                    connected.add(to_component)
                    unconnected.remove(to_component)
                    num_relations += 1
                except ValueError:
                    # Relation failed, try again
                    continue

            # Also add some relations between already connected components
            elif len(connected) >= 2:
                comp1, comp2 = random.sample(list(connected), 2)
                # Uniform random direction selection
                direction = random.choice(self.spatial_relations)

                try:
                    spatial_map.add_relation(comp1, comp2, direction)
                    num_relations += 1
                except ValueError:
                    # Relation already exists or conflicts, that's okay
                    continue

        # Ensure all components are connected
        # If some components remain unconnected, connect them
        while unconnected:
            from_component = random.choice(list(connected))
            to_component = unconnected.pop()

            # Try different directions until one works (use shuffle instead of sample)
            directions = self.spatial_relations.copy()
            random.shuffle(directions)
            for direction in directions:
                try:
                    spatial_map.add_relation(to_component, from_component, direction)
                    connected.add(to_component)
                    num_relations += 1
                    break
                except ValueError:
                    continue

        # Continue adding relations until target is reached (for uniqueness requirements)
        # This is important for stratified balancing where large structures need high relation density
        # Use much higher multiplier for large targets (needed for 8-component uniqueness)
        max_additional_attempts = (
            target_hops * 10 if target_hops > 12 else target_hops * 3
        )
        additional_attempts = 0
        while (
            num_relations < target_hops
            and additional_attempts < max_additional_attempts
        ):
            additional_attempts += 1

            # Add relations between already connected components
            if len(connected) >= 2:
                comp1, comp2 = random.sample(list(connected), 2)
                direction = random.choice(self.spatial_relations)

                try:
                    spatial_map.add_relation(comp1, comp2, direction)
                    num_relations += 1
                except ValueError:
                    # Relation already exists or conflicts, continue trying
                    continue

        return spatial_map, components, num_relations

    def _get_relation_component(self, relation: str, query_type: str) -> str:
        """Extract the relevant component of a relation based on query type.

        Args:
            relation: Full spatial relation (e.g., "upper-left", "above", "right")
            query_type: One of "full", "vertical", "horizontal"

        Returns:
            Relevant component for the query type
        """
        if query_type == "full":
            return relation
        elif query_type == "vertical":
            # Import here to avoid circular dependency
            try:
                from .spatial_utils import extract_vertical_component
            except ImportError:
                from spatial_utils import extract_vertical_component
            return extract_vertical_component(relation)
        else:  # horizontal
            try:
                from .spatial_utils import extract_horizontal_component
            except ImportError:
                from spatial_utils import extract_horizontal_component
            return extract_horizontal_component(relation)

    def _get_valid_query_types(
        self, spatial_map: SpatialMap, comp1: str, comp2: str
    ) -> List[str]:
        """Determine which query types are uniquely inferable for a component pair.

        Args:
            spatial_map: The spatial map containing the components
            comp1: First component
            comp2: Second component

        Returns:
            List of valid query types ("full", "vertical", "horizontal")
        """
        valid_types = []
        if spatial_map.is_relation_uniquely_inferable(comp1, comp2):
            valid_types.append("full")
        if spatial_map.is_vertical_relation_uniquely_inferable(comp1, comp2):
            valid_types.append("vertical")
        if spatial_map.is_horizontal_relation_uniquely_inferable(comp1, comp2):
            valid_types.append("horizontal")
        return valid_types

    def _find_valid_query_pairs(
        self, spatial_map: SpatialMap, components: List[str]
    ) -> List[Tuple[str, str, List[str]]]:
        """Find all component pairs with at least one valid query type.

        Args:
            spatial_map: The spatial map containing the components
            components: List of component names

        Returns:
            List of (comp1, comp2, valid_query_types) tuples
        """
        pair_options = []
        for i, comp1 in enumerate(components):
            for comp2 in components[i + 1 :]:
                valid_types = self._get_valid_query_types(spatial_map, comp1, comp2)
                if valid_types:
                    pair_options.append((comp1, comp2, valid_types))
        return pair_options

    def _select_query_pair(
        self,
        pair_options: List[Tuple[str, str, List[str]]],
        preferred_query_type: Optional[str],
    ) -> Tuple[str, str, str]:
        """Select component pair and query type based on preference.

        Args:
            pair_options: List of (comp1, comp2, valid_query_types) tuples
            preferred_query_type: Preferred query type if any

        Returns:
            Tuple of (comp1, comp2, selected_query_type)
        """
        if preferred_query_type:
            # Filter pairs that support the preferred type
            preferred_pairs = [
                (c1, c2, vt)
                for c1, c2, vt in pair_options
                if preferred_query_type in vt
            ]
            if preferred_pairs:
                comp1, comp2, valid_types = random.choice(preferred_pairs)
                return comp1, comp2, preferred_query_type

        # No preference or preferred type not available
        comp1, comp2, valid_types = random.choice(pair_options)
        return comp1, comp2, random.choice(valid_types)

    def generate_query(
        self,
        spatial_map: SpatialMap,
        components: List[str],
        preferred_query_type: Optional[str] = None,
    ) -> Optional[Tuple[str, str, str, bool, str]]:
        """
        Generate a query for the spatial structure.
        Intelligently selects query type based on which relations are uniquely inferable.

        For each pair of components, checks all three query types (full, vertical, horizontal).
        If a pair has multiple valid query types, randomly selects one to maintain balanced distribution.
        This maximizes data utilization by using pairs that may not be fully uniquely inferable
        but are uniquely inferable in vertical or horizontal dimensions.

        Args:
            spatial_map: The spatial map containing the components
            components: List of component names
            preferred_query_type: If specified, try to generate this query type first (for balancing)

        Returns:
            Tuple of (comp1, comp2, relation, is_direct, query_type) if valid query exists, None otherwise
        """
        # Find all component pairs with at least one valid query type
        pair_options = self._find_valid_query_pairs(spatial_map, components)
        if not pair_options:
            return None

        # Select component pair and query type
        comp1, comp2, query_type = self._select_query_pair(
            pair_options, preferred_query_type
        )

        # Get the actual relation based on selected query type
        if query_type == "vertical":
            relation = spatial_map.get_vertical_relation(comp1, comp2)
        elif query_type == "horizontal":
            relation = spatial_map.get_horizontal_relation(comp1, comp2)
        else:  # "full"
            relation = spatial_map.get_relation(comp1, comp2)

        # If filtering is enabled, check if relation is same level or same column
        # If so, try again by recursively calling generate_query (with a depth limit via return None)
        if self.exclude_same_level_column:
            if relation in ["same level", "same column"]:
                # Filter out this pair by removing it from consideration
                # Try to find another valid pair/query by returning None and letting caller retry
                return None

        # Check if this is a direct relation by looking at original stated relations
        # For full queries: exact relation must be stated
        # For vertical/horizontal queries: any stated relation with matching component counts as direct
        #   (e.g., "upper-left" stated → direct for both vertical:"above" and horizontal:"left")
        is_direct = any(
            (
                (
                    comp1 == obj1
                    and comp2 == obj2
                    and self._get_relation_component(rel, query_type) == relation
                )
                or (
                    comp1 == obj2
                    and comp2 == obj1
                    and self._get_relation_component(
                        spatial_map.INVERSE_DIRECTIONS.get(rel, rel), query_type
                    )
                    == relation
                )
            )
            for obj1, obj2, rel in spatial_map.stated_relations
        )

        return comp1, comp2, relation, is_direct, query_type

    def generate_instance(
        self,
        desired_terminology: Optional[str] = None,
        target_num_components: Optional[int] = None,
        target_num_relations: Optional[int] = None,
        preferred_query_type: Optional[str] = None,
        images_dir: Optional[str] = None,
        instance_id: Optional[str] = None,
    ) -> Dict:
        """Generate a single instance with all required fields.

        Args:
            desired_terminology: If specified, attempts to generate description with this terminology.
                               If None, uses random terminology (default: "spatial+cardinal+clock")
            target_num_components: If specified, generate with this number of components (for balancing)
            target_num_relations: If specified, aim for this number of relations (for balancing)
            preferred_query_type: If specified, try to generate this query type (for balancing)
            images_dir: If specified, generate JPG image and save to this directory
            instance_id: ID to use for naming the JPG file (required if images_dir is provided)

        Returns:
            Instance dictionary or None if generation failed
        """
        # Generate spatial structure with targeted parameters
        spatial_map, components, num_relations = self.generate_connected_structure(
            target_num_components=target_num_components,
            target_num_relations=target_num_relations,
        )

        # Select description mode based on desired terminology
        if desired_terminology and desired_terminology in self.TERMINOLOGY_TO_MODE:
            mode = self.TERMINOLOGY_TO_MODE[desired_terminology]
        else:
            # Default: use all available terminology
            mode = "spatial+cardinal+clock"

        # Generate natural language description with selected mode
        description, uses_cardinal, uses_clock = spatial_map.describe_relations(
            mode=mode
        )

        # Generate query with intelligent query type selection
        # The generate_query method will automatically choose the best query type
        # based on what's uniquely inferable for each pair
        # If preferred_query_type is specified, try to generate that type
        query_result = self.generate_query(
            spatial_map, components, preferred_query_type=preferred_query_type
        )
        if query_result is None:
            return None

        comp1, comp2, ground_truth, is_direct, query_type = query_result

        # Generate appropriate query string based on type
        if query_type == "vertical":
            query_relation = f"Where is {comp1} vertically relative to {comp2}?"
        elif query_type == "horizontal":
            query_relation = f"Where is {comp1} horizontally relative to {comp2}?"
        else:
            query_relation = f"Where is {comp1} relative to {comp2}?"

        # Generate ASCII visualizations
        try:
            ascii_simple = spatial_map.render("simple")
            ascii_grid = spatial_map.render("grid")
            ascii_panel = spatial_map.render("panel")
        except ValueError as e:
            # Should not happen as we ensure connectivity, but just in case
            print(f"Warning: Failed to render spatial map: {e}")
            return None

        # Get detailed relations and reasoning chain
        detailed_relations = spatial_map.get_detailed_relations()
        reasoning_steps = spatial_map.get_reasoning_steps(comp1, comp2)

        # Check if the spatial arrangement has a unique representation
        is_unique = spatial_map.has_unique_representation()

        # Get ambiguity tracking information
        ambiguity_count = spatial_map.ambiguity_count

        # Determine terminology used: check what actually appears in description
        # Spatial (base) terms can appear alongside cardinal/clock, so we check all three
        has_spatial = self._has_spatial_terms(description)

        # Build terminology label from combinations that appear
        terms = []
        if has_spatial:
            terms.append("spatial")
        if uses_cardinal:
            terms.append("cardinal")
        if uses_clock:
            terms.append("clock")

        terminology_used = "+".join(terms) if terms else "spatial"

        # Create instance dictionary
        instance = {
            "description": description,
            "query_relation": query_relation,
            "query_type": query_type,
            "label": ground_truth,
            "reasoning_steps": reasoning_steps,
            "detailed_relations": detailed_relations,
            "ascii": {"simple": ascii_simple, "grid": ascii_grid, "panel": ascii_panel},
            "terminology_used": terminology_used,  # Terminology in description: spatial, cardinal, clock, spatial+cardinal, spatial+clock, cardinal+clock, or spatial+cardinal+clock
            "num_components": len(components),
            "num_relations": num_relations,
            "is_directly_stated": is_direct,
            "has_unique_layout": is_unique,
            "ambiguous_stages": ambiguity_count,
        }

        # Generate JPG image if images_dir is provided
        if images_dir and instance_id:
            import os

            jpg_filename = os.path.join(images_dir, f"{instance_id}.jpg")
            try:
                spatial_map.generate_jpg(
                    filename=jpg_filename,
                    quality=100,
                    show_grid=True,
                    show_arrows=False,
                )
            except Exception as e:
                print(f"Warning: Failed to generate JPG for {instance_id}: {e}")

        return instance

    def _compute_dataset_statistics(
        self, instances: List[Dict], include_stratified: bool = False
    ) -> Dict:
        """Compute comprehensive dataset statistics in a single pass.

        Args:
            instances: List of generated instances
            include_stratified: If True, compute stratified uniqueness stats by component count

        Returns:
            Dictionary containing all computed statistics
        """
        stats = {
            "total": len(instances),
            "direct_count": 0,
            "unique_count": 0,
            "total_ambiguous_stages": 0,
            "total_components": 0,
            "total_relations": 0,
            "query_counts": {"full": 0, "vertical": 0, "horizontal": 0},
            "terminology_counts": {},
        }

        # Initialize stratified tracking if requested
        if include_stratified:
            stats["stratified_unique"] = {}

        for inst in instances:
            if inst["is_directly_stated"]:
                stats["direct_count"] += 1
            if inst["has_unique_layout"]:
                stats["unique_count"] += 1

            stats["total_ambiguous_stages"] += inst["ambiguous_stages"]
            stats["query_counts"][inst["query_type"]] += 1
            stats["total_components"] += inst["num_components"]
            stats["total_relations"] += inst["num_relations"]

            term_type = inst["terminology_used"]
            stats["terminology_counts"][term_type] = (
                stats["terminology_counts"].get(term_type, 0) + 1
            )

            # Track stratified uniqueness by component count
            if include_stratified:
                num_comp = inst["num_components"]
                if num_comp not in stats["stratified_unique"]:
                    stats["stratified_unique"][num_comp] = {
                        "unique": 0,
                        "non_unique": 0,
                        "total": 0,
                    }
                stats["stratified_unique"][num_comp]["total"] += 1
                if inst["has_unique_layout"]:
                    stats["stratified_unique"][num_comp]["unique"] += 1
                else:
                    stats["stratified_unique"][num_comp]["non_unique"] += 1

        return stats

    def _format_statistics(self, stats: Dict) -> str:
        """Format statistics as a readable string.

        Args:
            stats: Statistics dictionary from _compute_dataset_statistics

        Returns:
            Formatted statistics string
        """
        total = stats["total"]
        lines = ["Dataset Statistics:", f"  Total instances: {total}", "  Query types:"]

        # Query type statistics
        for qtype in ["full", "vertical", "horizontal"]:
            count = stats["query_counts"][qtype]
            pct = count / total * 100
            lines.append(
                f"    - {qtype.capitalize()} spatial queries: {count} ({pct:.1f}%)"
            )

        # Terminology statistics
        lines.append("  Terminology used in descriptions:")
        sorted_terminology = sorted(
            stats["terminology_counts"].items(), key=lambda x: -x[1]
        )
        for term_type, count in sorted_terminology:
            pct = count / total * 100
            lines.append(f"    - {term_type}: {count} ({pct:.1f}%)")

        # Direct/indirect statistics
        direct_count = stats["direct_count"]
        indirect_count = total - direct_count
        lines.extend(
            [
                f"  Direct relations: {direct_count} ({direct_count/total*100:.1f}%)",
                f"  Indirect relations: {indirect_count} ({indirect_count/total*100:.1f}%)",
            ]
        )

        # Uniqueness statistics
        unique_count = stats["unique_count"]
        non_unique_count = total - unique_count
        avg_ambiguous = stats["total_ambiguous_stages"] / total
        lines.extend(
            [
                f"  Unique representations: {unique_count} ({unique_count/total*100:.1f}%)",
                f"  Non-unique representations: {non_unique_count} ({non_unique_count/total*100:.1f}%)",
                f"  Total ambiguous stages: {stats['total_ambiguous_stages']}",
                f"  Average ambiguous stages per instance: {avg_ambiguous:.2f}",
            ]
        )

        # Average statistics
        avg_components = stats["total_components"] / total
        avg_relations = stats["total_relations"] / total
        lines.extend(
            [
                f"  Average components: {avg_components:.2f}",
                f"  Average relations: {avg_relations:.2f}",
            ]
        )

        # Stratified uniqueness statistics (if available)
        if "stratified_unique" in stats and stats["stratified_unique"]:
            lines.append("\n  Stratified Uniqueness by Component Count:")
            for num_comp in sorted(stats["stratified_unique"].keys()):
                s = stats["stratified_unique"][num_comp]
                unique_ratio = s["unique"] / s["total"] if s["total"] > 0 else 0
                lines.append(
                    f"    {num_comp} components: {s['unique']}/{s['total']} unique ({unique_ratio*100:.1f}%)"
                )

        return "\n".join(lines)

    def _select_generation_parameters(
        self,
        tracker: BalancingTracker,
        balance_terminology: bool,
        balance_components: bool,
        balance_relations: bool,
        balance_query_types: bool,
        stratify_unique_by_components: bool,
    ) -> Dict:
        """Select optimal generation parameters based on current balance state.

        Args:
            tracker: The balancing tracker with current statistics
            balance_terminology: Whether to balance terminology distribution
            balance_components: Whether to balance component counts
            balance_relations: Whether to balance relation counts
            balance_query_types: Whether to balance query types
            stratify_unique_by_components: Whether stratified balancing is enabled

        Returns:
            Dictionary with selected parameters (desired_terminology, target_components, etc.)
        """
        params = {}
        warmup_period = tracker._get_warmup_period()

        # Select terminology
        if balance_terminology and tracker.total_instances >= warmup_period:
            params["desired_terminology"] = tracker.get_most_needed_category(
                "terminology"
            )
            if params["desired_terminology"] is None:
                params["desired_terminology"] = random.choice(self.ALL_TERMINOLOGIES)

        # Select number of components
        if balance_components and tracker.total_instances >= warmup_period:
            target_components = tracker.get_most_needed_category("num_components")
            if isinstance(target_components, str):
                target_components = int(target_components)
            params["target_components"] = target_components

        # Adaptive relation targeting for stratified uniqueness balancing
        if (
            stratify_unique_by_components
            and params.get("target_components") is not None
        ):
            num_comp = params["target_components"]

            # Check if this stratum needs unique or non-unique layouts
            if (
                tracker.stratified_unique_counts
                and num_comp in tracker.stratified_unique_counts
            ):

                stratum = tracker.stratified_unique_counts[num_comp]
                if stratum["total"] >= 10:  # Need enough data
                    target_unique_ratio = tracker.targets.get("unique", 0.5)
                    current_unique_ratio = stratum["unique"] / stratum["total"]

                    # Calculate relation density needed for uniqueness
                    # For n components, there are n*(n-1)/2 possible relations
                    max_possible_relations = num_comp * (num_comp - 1) // 2

                    if current_unique_ratio < target_unique_ratio:
                        # Need MORE unique layouts → use MORE relations
                        # Aim for 70-90% coverage for uniqueness
                        min_coverage = 0.70
                        max_coverage = 0.90
                        target_rels = int(
                            max_possible_relations
                            * random.uniform(min_coverage, max_coverage)
                        )
                        # For stratified balancing, allow exceeding max_hops to achieve uniqueness
                        # But cap at the maximum possible relations
                        target_rels = max(
                            self.min_hops, min(max_possible_relations, target_rels)
                        )
                        params["target_relations"] = target_rels
                    elif current_unique_ratio > target_unique_ratio:
                        # Need MORE non-unique layouts → use FEWER relations
                        # Aim for 30-50% coverage for ambiguity
                        min_coverage = 0.30
                        max_coverage = 0.50
                        target_rels = int(
                            max_possible_relations
                            * random.uniform(min_coverage, max_coverage)
                        )
                        target_rels = max(
                            self.min_hops, min(self.max_hops, target_rels)
                        )
                        params["target_relations"] = target_rels

        # Standard relation selection (if not already set by adaptive logic)
        if (
            "target_relations" not in params
            and balance_relations
            and tracker.total_instances >= warmup_period
        ):
            target_relations = tracker.get_most_needed_category("num_relations")
            if isinstance(target_relations, str):
                target_relations = int(target_relations)
            params["target_relations"] = target_relations

        # Select query type
        if balance_query_types and tracker.total_instances >= warmup_period:
            params["preferred_query_type"] = tracker.get_most_needed_category(
                "query_type"
            )

        return params

    def generate_dataset(
        self,
        num_instances: int,
        output_file: str,
        target_unique_ratio: float = 0.5,
        target_direct_ratio: float = 0.5,
        balance_components: bool = True,
        balance_query_types: bool = True,
        balance_relations: bool = True,
        balance_terminology: bool = True,
        stratify_unique_by_components: bool = False,
        save_sample: bool = True,
        generate_images: bool = True,
    ):
        """Generate dataset with multi-dimensional balancing.

        Args:
            num_instances: Number of instances to generate
            output_file: Path to output JSONL file
            target_unique_ratio: Target ratio of unique representations (default: 0.5)
            target_direct_ratio: Target ratio of direct relations (default: 0.5)
            balance_components: Balance component count distribution (default: True)
            balance_query_types: Balance query type distribution (default: True)
            balance_relations: Balance relation count distribution (default: True)
            balance_terminology: Balance terminology distribution (default: True)
            stratify_unique_by_components: Balance uniqueness separately per component count (default: False)
            save_sample: Save first instance to sample_instance.json (default: True)
            generate_images: Generate JPG images for each instance (default: True)
        """
        # Initialize balancing tracker with all dimensions
        num_component_bins = self.max_components - self.min_components + 1
        num_relation_bins = self.max_hops - self.min_hops + 1

        targets = {"unique": target_unique_ratio, "direct": target_direct_ratio}

        num_bins = {}
        category_values = {}
        category_targets = {}

        if balance_query_types:
            num_bins["query_type"] = 3  # full, vertical, horizontal
            category_values["query_type"] = ["full", "vertical", "horizontal"]

        if balance_components:
            num_bins["num_components"] = num_component_bins
            # Provide all possible component counts
            category_values["num_components"] = list(
                range(self.min_components, self.max_components + 1)
            )

            # Weighted component distribution: prioritize complex layouts
            # 2 components: 2.5% (500 instances in 20k) - limited diversity
            # 3 components: 7.5% (1500 instances in 20k) - moderate diversity
            # 4-8 components: 18% each (3600 instances each in 20k) - high diversity
            component_weights = {
                2: 0.025,
                3: 0.075,
                4: 0.18,
                5: 0.18,
                6: 0.18,
                7: 0.18,
                8: 0.18,
            }
            category_targets["num_components"] = component_weights

        # Disable categorical relation balancing if using stratified uniqueness
        # because adaptive relation targeting handles relation counts intelligently
        if balance_relations and not stratify_unique_by_components:
            num_bins["num_relations"] = num_relation_bins
            # Provide all possible relation counts
            category_values["num_relations"] = list(
                range(self.min_hops, self.max_hops + 1)
            )

        if balance_terminology:
            num_bins["terminology"] = len(
                self.ALL_TERMINOLOGIES
            )  # 7 terminology combinations
            category_values["terminology"] = self.ALL_TERMINOLOGIES

        tracker = BalancingTracker(
            targets,
            num_bins,
            category_values,
            category_targets,
            stratify_unique_by_components=stratify_unique_by_components,
        )

        successful_instances = []
        attempts = 0
        rejected_count = 0
        # Adaptive max attempts based on number of balancing dimensions
        # Increased multiplier to accommodate stronger balancing
        # Stratified balancing needs even more attempts due to adaptive relation targeting
        if stratify_unique_by_components:
            base_multiplier = 40
        elif num_bins:
            base_multiplier = 25
        else:
            base_multiplier = 12
        max_attempts = num_instances * base_multiplier

        print(
            f"Generating {num_instances} spatial data instances with multi-dimensional balancing..."
        )
        print(
            f"  Targets: {target_unique_ratio*100:.0f}% unique, {target_direct_ratio*100:.0f}% direct"
        )
        if stratify_unique_by_components:
            print(
                f"  Stratified balancing: Uniqueness balanced separately per component count"
            )
            print(
                f"  Adaptive relation targeting: Adjusts relation density for uniqueness (may exceed max-hops)"
            )
        if num_bins:
            print(f"  Balancing: {', '.join(num_bins.keys())}")

        # Create images directory if image generation is enabled
        images_dir = None
        if generate_images:
            import os

            output_dir = os.path.dirname(os.path.abspath(output_file))
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            print(f"  Images will be saved to: {images_dir}")

        while len(successful_instances) < num_instances and attempts < max_attempts:
            attempts += 1

            # Proactively select parameters to minimize waste
            params = self._select_generation_parameters(
                tracker,
                balance_terminology,
                balance_components,
                balance_relations,
                balance_query_types,
                stratify_unique_by_components,
            )

            # Assign ID for this attempt (for JPG naming)
            instance_id = f"id-{len(successful_instances) + 1}"

            # Generate instance with targeted parameters (proactive, not reactive!)
            instance = self.generate_instance(
                desired_terminology=params.get("desired_terminology"),
                target_num_components=params.get("target_components"),
                target_num_relations=params.get("target_relations"),
                preferred_query_type=params.get("preferred_query_type"),
                images_dir=images_dir,
                instance_id=instance_id,
            )

            if instance is None:
                continue

            # Extract properties for balancing decision
            props = {
                "unique": instance["has_unique_layout"],
                "direct": instance["is_directly_stated"],
                "query_type": instance["query_type"],
                "num_components": instance["num_components"],
                "num_relations": instance["num_relations"],
                "terminology": instance["terminology_used"],
            }

            # Decide whether to accept based on current balance
            if not tracker.should_accept(**props):
                rejected_count += 1
                continue

            # Accept the instance
            instance["id"] = instance_id
            successful_instances.append(instance)
            tracker.update(**props)

            # Save the first instance as a sample
            if save_sample and len(successful_instances) == 1:
                import os

                output_dir = os.path.dirname(os.path.abspath(output_file))
                sample_file = os.path.join(output_dir, "sample_instance.json")
                os.makedirs(output_dir, exist_ok=True)
                with open(sample_file, "w", encoding="utf-8") as sf:
                    sf.write(json.dumps(instance, indent=2, ensure_ascii=False))
                print(f"  Sample instance saved to {sample_file}")

            # Progress updates with balance status
            if len(successful_instances) % 500 == 0:
                stats = tracker.get_stats()
                acceptance_rate = len(successful_instances) / attempts * 100
                print(
                    f"  Generated {len(successful_instances)}/{num_instances} instances (acceptance: {acceptance_rate:.1f}%)..."
                )
                print(
                    f"    Balance: Unique={stats['unique']['ratio']*100:.1f}% "
                    + f"Direct={stats['direct']['ratio']*100:.1f}%"
                )

                # Show stratified stats if enabled
                if stratify_unique_by_components and "stratified_unique" in stats:
                    strat_summary = []
                    for num_comp in sorted(stats["stratified_unique"].keys()):
                        s = stats["stratified_unique"][num_comp]
                        strat_summary.append(
                            f"{num_comp}C:{s['unique_ratio']*100:.0f}%"
                        )
                    print(f"    Stratified (unique%): {' '.join(strat_summary)}")

        # Warn if we didn't reach the target
        if len(successful_instances) < num_instances:
            acceptance_rate = len(successful_instances) / attempts * 100
            print(
                f"\nWarning: Only generated {len(successful_instances)}/{num_instances} instances "
                + f"(acceptance rate: {acceptance_rate:.1f}%)"
            )
            print(
                f"  Consider reducing balancing constraints or increasing max-attempts multiplier"
            )

        generation_summary = (
            f"Successfully generated {len(successful_instances)} instances."
        )
        print(generation_summary)

        with open(output_file, "w", encoding="utf-8") as f:
            for instance in successful_instances:
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

        data_saved_msg = f"Data saved to {output_file}"
        print(data_saved_msg)

        # Compute and format statistics
        stats = self._compute_dataset_statistics(
            successful_instances, include_stratified=stratify_unique_by_components
        )
        stats_text = self._format_statistics(stats)

        # Print to console
        print(f"\n{stats_text}")

        # Save statistics to file with generation summary
        stats_file = output_file.replace(".jsonl", "_stats.txt")
        full_stats_text = f"{generation_summary}\n{data_saved_msg}\n\n{stats_text}"
        with open(stats_file, "w", encoding="utf-8") as f:
            f.write(full_stats_text)

        print(f"\nAll output information has been saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spatial layout data with multi-dimensional balancing"
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=8,
        help="Maximum number of spatial components (default: 8)",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=12,
        help="Maximum number of relational hops (default: 12)",
    )
    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Minimum number of spatial components (default: 2)",
    )
    parser.add_argument(
        "--min-hops",
        type=int,
        default=1,
        help="Minimum number of relational hops (default: 1)",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=20000,
        help="Number of instances to generate (default: 20000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/spatial_data.jsonl",
        help="Output JSONL file path (default: datasets/spatial_data.jsonl)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--exclude-same-level-column",
        action="store_true",
        help="Exclude same_level and same_column queries (default: False)",
    )

    # Balancing options
    parser.add_argument(
        "--target-unique-ratio",
        type=float,
        default=0.5,
        help="Target ratio of unique representations (default: 0.5 for 50%%)",
    )
    parser.add_argument(
        "--target-direct-ratio",
        type=float,
        default=0.5,
        help="Target ratio of direct relations (default: 0.5 for 50%%)",
    )
    parser.add_argument(
        "--stratify-unique-by-components",
        action="store_true",
        help="Balance uniqueness separately per component count (combats natural bias)",
    )
    parser.add_argument(
        "--no-balance-components",
        action="store_true",
        help="Disable component count balancing",
    )
    parser.add_argument(
        "--no-balance-query-types",
        action="store_true",
        help="Disable query type balancing",
    )
    parser.add_argument(
        "--no-balance-relations",
        action="store_true",
        help="Disable relation count balancing",
    )
    parser.add_argument(
        "--no-balance-terminology",
        action="store_true",
        help="Disable terminology distribution balancing",
    )
    parser.add_argument(
        "--no-generate-images",
        action="store_true",
        help="Disable JPG image generation (default: images are generated)",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    # Validate target ratios
    if not 0.0 <= args.target_unique_ratio <= 1.0:
        parser.error("--target-unique-ratio must be between 0.0 and 1.0")
    if not 0.0 <= args.target_direct_ratio <= 1.0:
        parser.error("--target-direct-ratio must be between 0.0 and 1.0")

    generator = SpatialDataGenerator(
        args.max_components,
        args.max_hops,
        args.min_components,
        args.min_hops,
        exclude_same_level_column=args.exclude_same_level_column,
    )

    generator.generate_dataset(
        args.num_instances,
        args.output,
        target_unique_ratio=args.target_unique_ratio,
        target_direct_ratio=args.target_direct_ratio,
        balance_components=not args.no_balance_components,
        balance_query_types=not args.no_balance_query_types,
        balance_relations=not args.no_balance_relations,
        balance_terminology=not args.no_balance_terminology,
        stratify_unique_by_components=args.stratify_unique_by_components,
        generate_images=not args.no_generate_images,
    )


if __name__ == "__main__":
    main()
