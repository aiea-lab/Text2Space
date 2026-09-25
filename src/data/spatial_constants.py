"""
Shared constants for spatial relationship processing.

This module contains all direction mappings, synonyms, and label aliases
used across the spatial reasoning codebase.
"""

from typing import Dict, List

# ============================================================================
# CORE DIRECTION MAPPINGS
# ============================================================================

# Base directional relationships (8 directions)
BASE_DIRECTIONS = [
    "above",
    "below",
    "left",
    "right",
    "upper-left",
    "upper-right",
    "lower-left",
    "lower-right",
]

# Inverse direction mappings for bidirectional relationships
INVERSE_DIRECTIONS = {
    "above": "below",
    "below": "above",
    "left": "right",
    "right": "left",
    "upper-left": "lower-right",
    "upper-right": "lower-left",
    "lower-left": "upper-right",
    "lower-right": "upper-left",
}

# Position offsets for each direction (dx, dy)
# Note: y increases downward (consistent with ASCII grid row indices)
POSITION_OFFSETS = {
    "above": (0, -1),
    "below": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
    "upper-left": (-1, -1),
    "upper-right": (1, -1),
    "lower-left": (-1, 1),
    "lower-right": (1, 1),
}

# ============================================================================
# NATURAL LANGUAGE GENERATION
# ============================================================================

# Spatial direction synonyms for natural language generation
# All phrases work in the format: "A is {RELATION} B"
DIRECTION_SYNONYMS = {
    "above": [
        "above",
        "over",
        "higher than",
        "on top of",
        "positioned above",
        "located above",
    ],
    "below": [
        "below",
        "under",
        "beneath",
        "lower than",
        "underneath",
        "positioned below",
        "located below",
    ],
    "left": [
        "to the left of",
        "on the left side of",
        "left of",
        "positioned to the left of",
        "located to the left of",
    ],
    "right": [
        "to the right of",
        "on the right side of",
        "right of",
        "positioned to the right of",
        "located to the right of",
    ],
    "upper-left": [
        "above and to the left of",
        "to the upper-left of",
        "at the top-left of",
        "in the upper-left of",
        "positioned upper-left of",
        "to the left and above",
        "left and above",
        "above and left",
        "to the above and to the left of",
        "to the left and to the above of",
        "to the left of and above",
        "to the above of and left",
    ],
    "upper-right": [
        "above and to the right of",
        "to the upper-right of",
        "at the top-right of",
        "in the upper-right of",
        "positioned upper-right of",
        "to the right and above",
        "right and above",
        "above and right",
        "to the above and to the right of",
        "to the right and to the above of",
        "to the right of and above",
        "to the above of and right",
    ],
    "lower-left": [
        "below and to the left of",
        "to the lower-left of",
        "at the bottom-left of",
        "in the lower-left of",
        "positioned lower-left of",
        "to the left and below",
        "left and below",
        "below and left",
        "to the below and to the left of",
        "to the left and to the below of",
        "to the left of and below",
        "to the below of and left",
    ],
    "lower-right": [
        "below and to the right of",
        "to the lower-right of",
        "at the bottom-right of",
        "in the lower-right of",
        "positioned lower-right of",
        "to the right and below",
        "right and below",
        "below and right",
        "to the below and to the right of",
        "to the right and to the below of",
        "to the right of and below",
        "to the below of and right",
    ],
}

# Cardinal direction synonyms for natural language generation
# All phrases work in the format: "A is {RELATION} B"
CARDINAL_SYNONYMS = {
    "above": [
        "north of",
        "to the north of",
        "directly north of",
        "northward of",
        "northward from",
    ],
    "below": [
        "south of",
        "to the south of",
        "directly south of",
        "southward of",
        "southward from",
    ],
    "left": [
        "west of",
        "to the west of",
        "directly west of",
        "westward of",
        "westward from",
    ],
    "right": [
        "east of",
        "to the east of",
        "directly east of",
        "eastward of",
        "eastward from",
    ],
    "upper-left": [
        "to the northwest of",
        "northwest of",
        "north and west of",
        "to the north and west of",
        "west and north of",
        "to the west and north of",
        "to the north and to the west of",
        "to the west and to the north of",
        "northwestward of",
        "northwestward from",
    ],
    "upper-right": [
        "to the northeast of",
        "northeast of",
        "north and east of",
        "to the north and east of",
        "east and north of",
        "to the east and north of",
        "to the north and to the east of",
        "to the east and to the north of",
        "northeastward of",
        "northeastward from",
    ],
    "lower-left": [
        "to the southwest of",
        "southwest of",
        "south and west of",
        "to the south and west of",
        "west and south of",
        "to the west and south of",
        "to the south and to the west of",
        "to the west and to the south of",
        "southwestward of",
        "southwestward from",
    ],
    "lower-right": [
        "to the southeast of",
        "southeast of",
        "south and east of",
        "to the south and east of",
        "east and south of",
        "to the east and south of",
        "to the south and to the east of",
        "to the east and to the south of",
        "southeastward of",
        "southeastward from",
    ],
}

# Clock position synonyms for natural language generation
# All phrases work in the format: "A is {RELATION} B"
CLOCK_SYNONYMS = {
    "above": [
        "at 12 o'clock from",
        "at the 12 o'clock position from",
        "at 12:00 from",
        "at the 12:00 position from",
        "at 0 o'clock from",
        "at the 0 o'clock position from",
    ],
    "below": [
        "at 6 o'clock from",
        "at the 6 o'clock position from",
        "at 6:00 from",
        "at the 6:00 position from",
    ],
    "left": [
        "at 9 o'clock from",
        "at the 9 o'clock position from",
        "at 9:00 from",
        "at the 9:00 position from",
    ],
    "right": [
        "at 3 o'clock from",
        "at the 3 o'clock position from",
        "at 3:00 from",
        "at the 3:00 position from",
    ],
    "upper-left": [
        "at 10 o'clock from",
        "at the 10 o'clock position from",
        "at 10:30 from",
        "at the 10:30 position from",
        "at 11 o'clock from",
        "at the 11 o'clock position from",
    ],
    "upper-right": [
        "at 1 o'clock from",
        "at the 1 o'clock position from",
        "at 1:30 from",
        "at the 1:30 position from",
        "at 2 o'clock from",
        "at the 2 o'clock position from",
    ],
    "lower-left": [
        "at 7 o'clock from",
        "at the 7 o'clock position from",
        "at 7:30 from",
        "at the 7:30 position from",
        "at 8 o'clock from",
        "at the 8 o'clock position from",
    ],
    "lower-right": [
        "at 4 o'clock from",
        "at the 4 o'clock position from",
        "at 4:30 from",
        "at the 4:30 position from",
        "at 5 o'clock from",
        "at the 5 o'clock position from",
    ],
}

# ============================================================================
# EVALUATION / MATCHING
# ============================================================================

# Comprehensive label aliases for evaluation/matching
# Includes all possible variations that models might generate as answers
LABEL_ALIASES = {
    # Full spatial directions (8 directions)
    "above": [
        "above",
        "over",
        "top",
        "upper",
        "higher",
        "north",
        "northern",
        "upward",
        "up",
        "12:00",
        "12 o'clock",
        "0:00",
        "0 o'clock",
        "positioned above",
        "located above",
        "higher than",
        "on top",
    ],
    "below": [
        "below",
        "under",
        "beneath",
        "bottom",
        "lower",
        "south",
        "southern",
        "downward",
        "down",
        "6:00",
        "6 o'clock",
        "positioned below",
        "located below",
        "lower than",
        "underneath",
    ],
    "left": [
        "left",
        "west",
        "western",
        "leftward",
        "9:00",
        "9 o'clock",
        "positioned to the left",
        "located to the left",
        "to the left",
        "on the left side",
        "left of",
        "to the west",
    ],
    "right": [
        "right",
        "east",
        "eastern",
        "rightward",
        "3:00",
        "3 o'clock",
        "positioned to the right",
        "located to the right",
        "to the right",
        "on the right side",
        "right of",
        "to the east",
    ],
    "upper-left": [
        "upper-left",
        "top-left",
        "upper left",
        "top left",
        "northwest",
        "northwestern",
        "10:30",
        "10 o'clock",
        "11 o'clock",
        "positioned upper-left",
        "located upper-left",
        "above and to the left",
        "to the upper-left",
        "at the top-left",
        "to the northwest",
        "in the upper-left",
        "north-west",  # hyphenated version
        "north west",  # space-separated version
        "northwestward",  # directional variant
    ],
    "upper-right": [
        "upper-right",
        "top-right",
        "upper right",
        "top right",
        "northeast",
        "northeastern",
        "1:30",
        "1 o'clock",
        "2 o'clock",
        "positioned upper-right",
        "located upper-right",
        "above and to the right",
        "to the upper-right",
        "at the top-right",
        "to the northeast",
        "in the upper-right",
        "north-east",  # hyphenated version not in main LABEL_ALIASES
        "north east",  # space-separated version
        "northeastward",  # directional variant
    ],
    "lower-left": [
        "lower-left",
        "bottom-left",
        "lower left",
        "bottom left",
        "southwest",
        "southwestern",
        "7:30",
        "7 o'clock",
        "8 o'clock",
        "positioned lower-left",
        "located lower-left",
        "below and to the left",
        "to the lower-left",
        "at the bottom-left",
        "to the southwest",
        "in the lower-left",
        "south-west",  # hyphenated version
        "south west",  # space-separated version
        "southwestward",  # directional variant
    ],
    "lower-right": [
        "lower-right",
        "bottom-right",
        "lower right",
        "bottom right",
        "southeast",
        "southeastern",
        "4:30",
        "4 o'clock",
        "5 o'clock",
        "positioned lower-right",
        "located lower-right",
        "below and to the right",
        "to the lower-right",
        "at the bottom-right",
        "to the southeast",
        "in the lower-right",
        "south-east",  # hyphenated version
        "south east",  # space-separated version
        "southeastward",  # directional variant
    ],
    # Vertical-only relations
    "same level": [
        "same level",
        "same height",
        "horizontally aligned",
        "same vertical position",
        "same y",
        "same row",
        "aligned horizontally",
        "level",
        "even",
    ],
    # Horizontal-only relations
    "same column": [
        "same column",
        "same horizontal position",
        "vertically aligned",
        "same x",
        "aligned vertically",
        "same vertical line",
        "inline",
    ],
    # Same position (shouldn't occur with current position logic, but included for completeness)
    "same position": [
        "same position",
        "same location",
        "same spot",
        "identical position",
        "co-located",
        "at the same position",
        "in the same location",
    ],
}

# Reverse mapping: synonym -> base direction (for parsing)
SYNONYM_TO_BASE: Dict[str, str] = {}
for base_dir, aliases in LABEL_ALIASES.items():
    for alias in aliases:
        SYNONYM_TO_BASE[alias.lower()] = base_dir

# ============================================================================
# QUERY TYPES
# ============================================================================

# Vertical relationship types
VERTICAL_RELATIONS = ["above", "below", "same level"]

# Horizontal relationship types
HORIZONTAL_RELATIONS = ["left", "right", "same column"]

# Query types for spatial reasoning
QUERY_TYPES = ["full_spatial", "vertical", "horizontal"]
