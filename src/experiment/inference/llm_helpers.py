"""
Helper functions for spatial reasoning LLM evaluation.
Clean implementation with separated prompts and comparison logic.
"""

import json
import re

ASCII_VIEW_DESCRIPTIONS = {
    "grid": "<grid using + - | for borders>",
    "simple": "<tiny freeform sketch>",
    "panel": "<box-drawing view>",
}

VALID_ASCII_VIEWS = ("grid", "simple", "panel")

TASK_CONFIGS: dict[str, dict[str, dict[str, bool]]] = {
    "A": {
        "inputs": {"description": True, "ascii": False, "query": False},
        "outputs": {"description": False, "ascii": True, "answer": False},
    },
    "B": {
        "inputs": {"description": False, "ascii": True, "query": False},
        "outputs": {"description": True, "ascii": False, "answer": False},
    },
    "C": {
        "inputs": {"description": True, "ascii": False, "query": True},
        "outputs": {"description": False, "ascii": False, "answer": True},
    },
    "D": {
        "inputs": {"description": False, "ascii": True, "query": True},
        "outputs": {"description": False, "ascii": False, "answer": True},
    },
    "E": {
        "inputs": {"description": True, "ascii": False, "query": True},
        "outputs": {"description": False, "ascii": True, "answer": True},
    },
    "F": {
        "inputs": {"description": True, "ascii": True, "query": True},
        "outputs": {"description": False, "ascii": False, "answer": True},
    },
    "G": {
        "inputs": {"description": True, "ascii": False, "query": True},
        "outputs": {"description": False, "ascii": True, "answer": False},
    },
    "H": {
        "inputs": {"description": True, "ascii": False, "query": True},
        "outputs": {"description": False, "ascii": True, "answer": False},
    },
}


def _list_enabled(flags: dict[str, bool]) -> list[str]:
    """Return the enabled keys in order."""
    return [key for key, enabled in flags.items() if enabled]


TASK_METADATA: dict[str, dict[str, object]] = {
    "A": {
        "title": "Description → ASCII Diagram",
        "objective": "Read the textual scene description and render the requested ASCII layout.",
        "input_notes": [
            "Use only the provided description to infer entity positions.",
            "Do not invent additional entities or relations when information is missing.",
        ],
        "output_notes": [
            "Produce ASCII views that faithfully match the described scene.",
        ],
        "ascii_guidelines": [
            "Label every entity exactly as named in the description.",
        ],
    },
    "B": {
        "title": "ASCII Diagram → Description",
        "objective": "Interpret the ASCII evidence and rewrite it as a concise natural-language description.",
        "input_notes": [
            "Treat the ASCII diagram as authoritative evidence.",
        ],
        "output_notes": [
            "Summarize the spatial arrangement in clear declarative sentences.",
        ],
        "description_guidelines": [
            "Mention every entity shown in the ASCII input and describe their relative placement.",
            "Keep the description short and factual; avoid speculative language.",
            "The spatial summary will consist of sentences in the format “A is [relation] B.” Each sentence should describe the relationship between exactly two objects.",
        ],
    },
    "C": {
        "title": "Description + Query → Answer",
        "objective": "Answer the spatial relationship question using the textual description.",
        "input_notes": [
            "Use the description to reason about the positions of the queried entities.",
        ],
        "output_notes": [
            "Return the spatial relation as a single direction string.",
        ],
        "direction_notes": [
            "Ensure the answer directly references the entities named in the question.",
        ],
    },
    "D": {
        "title": "ASCII + Query → Answer",
        "objective": "Answer the spatial relationship question using the ASCII diagram as the primary evidence.",
        "input_notes": [
            "Use the ASCII diagram as ground truth; the query specifies the entities to compare.",
        ],
        "output_notes": [
            "Provide a single directional answer from the defined set.",
        ],
        "direction_notes": [
            "Base the answer on the ASCII layout; ignore assumptions that contradict the diagram.",
        ],
    },
    "E": {
        "title": "Description + Query → ASCII + Answer",
        "objective": "Answer the spatial question and draw ASCII that matches the same layout.",
        "input_notes": [
            "Use the description to understand the scene before answering.",
            "Ensure the ASCII rendering is consistent with the textual evidence.",
        ],
        "output_notes": [
            "Provide both the answer and the requested ASCII views in the JSON output.",
        ],
        "ascii_guidelines": [
            "Ensure the ASCII visualization aligns with the answer you provide.",
        ],
        "direction_notes": [
            "Check the ASCII you generate so it supports the chosen answer.",
        ],
    },
    "F": {
        "title": "Description + ASCII + Query → Answer",
        "objective": "Combine textual and ASCII evidence to answer the spatial question.",
        "input_notes": [
            "Cross-check the description and ASCII diagram; treat ASCII as authoritative when conflicts arise.",
        ],
        "output_notes": [
            "Output a single canonical direction that matches the combined evidence.",
        ],
        "direction_notes": [
            "If the two evidence sources disagree, follow the ASCII diagram.",
        ],
    },
    "G": {
        "title": "Two-Turn Task: Generate ASCII then Answer",
        "objective": "This is a two-turn task:\n- Turn 1: Use the description and question to generate an ASCII visualization that will help answer the question.\n- Turn 2: Use the ASCII you generated in Turn 1 to answer the spatial question.",
        "input_notes": [
            "Turn 1: Use the description and question to generate an ASCII visualization that will help answer the question.",
            "Turn 2: Use the ASCII you generated in Turn 1 to answer the spatial question.",
        ],
        "output_notes": [
            'Turn 1: Provide the ASCII views as a JSON object with "ascii" key.',
            'Turn 2: Provide the directional answer as a JSON object with "answer" key.',
        ],
        "direction_notes": [
            "Base your answer on the ASCII visualization you generated in Turn 1.",
            "Treat the ASCII as the primary evidence when answering.",
        ],
    },
    "H": {
        "title": "Two-Turn Task: Generate ASCII then Answer (with Ground Truth ASCII)",
        "objective": "This is a two-turn task:\n- Turn 1: Use the description and question to generate an ASCII visualization that will help answer the question.\n- Turn 2: Use the provided ground truth ASCII diagram along with the description and question to answer the spatial question.",
        "input_notes": [
            "Turn 1: Use the description and question to generate an ASCII visualization that will help answer the question.",
            "Turn 2: Use the provided ground truth ASCII diagram along with the description and question to answer the spatial question.",
        ],
        "output_notes": [
            'Turn 1: Provide the ASCII views as a JSON object with "ascii" key.',
            'Turn 2: Provide the directional answer as a JSON object with "answer" key.',
        ],
        "direction_notes": [
            "Base your answer on the provided ground truth ASCII diagram.",
            "Treat the ASCII as the primary evidence when answering.",
        ],
    },
}

TASK_SUMMARY_AE = {
    code: {
        "title": TASK_METADATA[code]["title"],
        "objective": TASK_METADATA[code]["objective"],
        "inputs": _list_enabled(TASK_CONFIGS[code]["inputs"]),
        "outputs": _list_enabled(TASK_CONFIGS[code]["outputs"]),
    }
    for code in ("A", "B", "C", "D", "E")
}


def _normalize_ascii_views(selection) -> list[str]:
    """
    Normalize a string/list selection of ASCII views into an ordered, de-duplicated list.
    """
    if selection is None:
        return []
    if isinstance(selection, str):
        selection = [selection]
    resolved = []
    for view in selection:
        if view not in VALID_ASCII_VIEWS:
            continue
        if view not in resolved:
            resolved.append(view)
    return resolved


def _format_ascii_block(
    ascii_block, views: list[str] | None = None, indent_text: str = ""
) -> str:
    """
    Render an ASCII block (dict or str) into a printable string.
    """
    if not ascii_block:
        return ""
    if isinstance(ascii_block, str):
        lines = ascii_block.rstrip("\n")
        if indent_text:
            lines = "\n".join(f"{indent_text}{line}" for line in lines.splitlines())
        return lines

    views_to_show = views or list(ascii_block.keys())
    rendered_sections = []
    for view in views_to_show:
        art = ascii_block.get(view)
        if not art:
            continue
        block = art.rstrip("\n")
        if indent_text:
            block = "\n".join(f"{indent_text}{line}" for line in block.splitlines())
        rendered_sections.append(f"{view}:\n{block}")
    return "\n\n".join(rendered_sections)


def json_dumps_compact(obj):
    """Serialize JSON with minimal whitespace for compact prompts."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def format_few_shot_example(
    ex: dict,
    task_inputs: dict[str, bool],
    task_outputs: dict[str, bool],
    ascii_views: list[str],
    ascii_order: str,
    ascii_input_views: list[str] | None = None,
) -> str:
    """Format a single few-shot example following the task configuration."""
    input_sections: list[str] = []

    if task_inputs.get("description") and ex.get("description"):
        input_sections.append(f'Description="{ex["description"]}"')

    if task_inputs.get("ascii"):
        ascii_text = _format_ascii_block(ex.get("ascii"), views=ascii_input_views)
        if ascii_text:
            input_sections.append(f"ASCII:\n{ascii_text}")

    if task_inputs.get("query") and ex.get("query_relation"):
        input_sections.append(f'Query="{ex["query_relation"]}"')

    input_block = (
        "Input:\n" + "\n\n".join(input_sections)
        if input_sections
        else "Input: (none provided)"
    )

    ascii_block = ex.get("ascii", {})
    ascii_output = {}
    if task_outputs.get("ascii") and ascii_block:
        views_to_use = ascii_views or list(ascii_block.keys())
        for view in views_to_use:
            art = ascii_block.get(view)
            if art:
                ascii_output[view] = art

    answer_value = ex.get("label")
    description_value = (
        ex.get("description") if task_outputs.get("description") else None
    )

    output_items: list[tuple[str, object]] = []
    if description_value and task_outputs.get("description"):
        output_items.append(("description", description_value))

    ascii_entry = ("ascii", ascii_output) if ascii_output else None
    answer_entry = (
        ("answer", answer_value)
        if task_outputs.get("answer") and answer_value
        else None
    )

    if ascii_entry and answer_entry:
        if ascii_order == "ascii_first":
            output_items.extend([ascii_entry, answer_entry])
        else:
            output_items.extend([answer_entry, ascii_entry])
    else:
        if ascii_entry:
            output_items.append(ascii_entry)
        if answer_entry:
            output_items.append(answer_entry)

    output_obj = {key: value for key, value in output_items}
    output_block = json_dumps_compact(output_obj) if output_obj else '""'

    return f"{input_block}\nOutput: {output_block}"


def _build_output_schema(
    *,
    ascii_views: list[str],
    ascii_format_output: bool,
    include_answer: bool,
    include_description_output: bool,
    ascii_order: str,
) -> str:
    """Construct a JSON schema string to describe the expected model output."""
    fields: list[str] = []

    ascii_block = ""
    if ascii_format_output and ascii_views:
        ascii_fields = ",\n    ".join(
            f'"{view}": "{ASCII_VIEW_DESCRIPTIONS[view]}"' for view in ascii_views
        )
        ascii_block = f'"ascii": {{\n    {ascii_fields}\n  }}'

    if include_description_output:
        fields.append('"description": "<natural language spatial summary>"')

    if ascii_format_output and ascii_views:
        if include_answer:
            if ascii_order == "ascii_first":
                if ascii_block:
                    fields.append(ascii_block)
                fields.append('"answer": "<direction>"')
            else:
                fields.append('"answer": "<direction>"')
                if ascii_block:
                    fields.append(ascii_block)
        else:
            if ascii_block:
                fields.append(ascii_block)
    else:
        if include_answer:
            fields.append('"answer": "<direction>"')

    if (
        not include_answer
        and include_description_output
        and ascii_block
        and ascii_block not in fields
    ):
        fields.append(ascii_block)

    if not fields:
        return "{}"

    return "{\n  " + ",\n  ".join(fields) + "\n}"


_BASE_CARDINAL_SYNONYMS: dict[str, set[str]] = {
    "left": {"left", "left side", "to the left", "leftward", "west"},
    "right": {"right", "right side", "to the right", "rightward", "east"},
    "above": {
        "above",
        "up",
        "upwards",
        "north",
        "over",
        "overhead",
        "top",
        "upper",
        "higher",
    },
    "below": {
        "below",
        "down",
        "downwards",
        "south",
        "under",
        "beneath",
        "bottom",
        "lower",
    },
}

_DIAGONAL_COMPONENTS: dict[str, tuple[str, str]] = {
    "top-left": ("above", "left"),
    "top-right": ("above", "right"),
    "bottom-left": ("below", "left"),
    "bottom-right": ("below", "right"),
}

_DIAGONAL_EXTRAS: dict[str, set[str]] = {
    "top-left": {"northwest", "north west", "north-west"},
    "top-right": {"northeast", "north east", "north-east"},
    "bottom-left": {"southwest", "south west", "south-west"},
    "bottom-right": {"southeast", "south east", "south-east"},
}

_STATIC_DIRECTION_SYNONYMS: dict[str, set[str]] = {
    "overlap": {
        "overlap",
        "overlapping",
        "same spot",
        "same place",
        "same position",
        "same point",
        "same location",
        "coincident",
        "stacked",
    },
    "same spot": {
        "same spot",
        "same place",
        "same position",
        "same point",
        "same location",
        "overlap",
        "coincident",
    },
    "same column": {
        "same column",
        "same vertical line",
        "aligned vertically",
        "vertical alignment",
        "share column",
    },
    "same row": {
        "same row",
        "same horizontal line",
        "aligned horizontally",
        "horizontal alignment",
        "share row",
    },
    "same level": {
        "same level",
        "same height",
        "same elevation",
        "equal level",
        "even level",
    },
}

_DIRECTION_SYNONYMS: dict[str, set[str]] = {}
_DIRECTION_SYNONYMS.update({k: set(v) for k, v in _BASE_CARDINAL_SYNONYMS.items()})

for diagonal, (vertical, horizontal) in _DIAGONAL_COMPONENTS.items():
    combos: set[str] = set()
    vertical_forms = _BASE_CARDINAL_SYNONYMS.get(vertical, set()).union({vertical})
    horizontal_forms = _BASE_CARDINAL_SYNONYMS.get(horizontal, set()).union(
        {horizontal}
    )
    for v in vertical_forms:
        for h in horizontal_forms:
            combos.add(f"{v} {h}")
            combos.add(f"{v}-{h}")
            combos.add(f"{v}_{h}")
            combos.add(f"{v}{h}")
    combos.update(_DIAGONAL_EXTRAS.get(diagonal, set()))
    _DIRECTION_SYNONYMS[diagonal] = combos

for key, values in _STATIC_DIRECTION_SYNONYMS.items():
    _DIRECTION_SYNONYMS[key] = set(values)


def _normalize_direction(value: str) -> str:
    if not value:
        return ""
    text = value.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    text = text.replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    return text


def _generate_variants(term: str) -> set[str]:
    base = term.lower().strip()
    base = base.replace("_", " ").replace("-", " ")
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        return set()
    tokens = base.split()
    variants = {" ".join(tokens), "-".join(tokens), "_".join(tokens), "".join(tokens)}
    return variants


_CANONICAL_LOOKUP: dict[str, str] = {}
for canonical, synonyms in _DIRECTION_SYNONYMS.items():
    variants = set(synonyms)
    variants.add(canonical)
    expanded_variants = set()
    for variant in variants:
        expanded_variants.update(_generate_variants(variant))
    variants.update(expanded_variants)
    for variant in variants:
        normalized = _normalize_direction(variant)
        if normalized:
            _CANONICAL_LOOKUP[normalized] = canonical


def _canonicalize_direction(value: str) -> str:
    normalized = _normalize_direction(value)
    if not normalized:
        return ""
    key = normalized.replace(" ", "")
    if key in _CANONICAL_LOOKUP:
        return _CANONICAL_LOOKUP[key]
    return _CANONICAL_LOOKUP.get(normalized, normalized)


# ============================================================================
# EVALUATION FUNCTIONS - DEPRECATED
# ============================================================================
# These functions are commented out. Use ASCIIEvaluator from ascii_evaluator.py instead.
# The evaluator handles all standardization and comparison logic internally.

# def compare_results(predicted_answer, label_aliases):
#     """
#     Compare predicted answer against ground truth label and aliases.
#
#     Args:
#         predicted_answer: str - The model's predicted spatial relationship
#         label_aliases: list - List of valid aliases for the correct answer
#
#     Returns:
#         bool - True if prediction matches any alias (case-insensitive), False otherwise
#     """
#     predicted_canonical = _canonicalize_direction(predicted_answer)
#     predicted_plain = predicted_answer.strip().lower()
#
#     # Check if prediction matches any alias
#     for alias in label_aliases:
#         alias_canonical = _canonicalize_direction(alias)
#         if predicted_canonical and alias_canonical and predicted_canonical == alias_canonical:
#             return True
#         if predicted_plain == alias.lower().strip():
#             return True
#
#     return False


# def evaluate_direction(prediction: str, ground_truth: str) -> bool:
#     """
#     Compare a prediction and a ground-truth direction string using the same
#     canonicalization as compare_results.
#
#     Args:
#         prediction: str - Predicted spatial relationship.
#         ground_truth: str - Ground-truth spatial relationship.
#
#     Returns:
#         bool - True if both strings map to the same canonical direction.
#     """
#     return _canonicalize_direction(prediction) == _canonicalize_direction(ground_truth)


def rephrase_spatial_query(sentence: str) -> str:
    """
    Convert phrases like "What is the spatial relationship between A and F?"
    into "What's the position of A to F?"
    """
    sentence = sentence.strip()
    pattern = r"between\s+([A-Za-z])\s+and\s+([A-Za-z])"
    match = re.search(pattern, sentence, re.IGNORECASE)

    if match:
        first, second = match.groups()
        new_sentence = f"What's the position of {first} to {second}?"
        return new_sentence
    else:
        return "Could not identify entities to rephrase."


def build_system_prompt(
    task_code: str,
    *,
    ascii_format_input=None,
    ascii_format_output=None,
    include_answer: bool = True,
    include_description_output: bool = False,
    include_query: bool = True,
    simple_mode: bool = False,
    ascii_order: str = "answer_first",
) -> str:
    """
    Build system prompt tailored to a specific task configuration.
    """
    task_config = TASK_CONFIGS.get(task_code, TASK_CONFIGS["C"])
    task_inputs = task_config["inputs"]
    task_outputs = task_config["outputs"]
    task_meta = TASK_METADATA.get(task_code, TASK_METADATA["C"])

    direction_list = "{left, right, above, below, upper-left, upper-right, lower-left, lower-right, same column, same level}"

    input_views = _normalize_ascii_views(ascii_format_input)
    output_views = _normalize_ascii_views(ascii_format_output)

    task_requires_ascii_input = task_inputs.get("ascii", False)
    task_requires_ascii_output = task_outputs.get("ascii", False)

    if not task_requires_ascii_input:
        input_views = []
    if not task_requires_ascii_output:
        output_views = []

    if task_requires_ascii_input and not input_views:
        input_views = ["grid"]

    if task_requires_ascii_output and not output_views:
        output_views = ["grid"]

    has_ascii_input = bool(input_views)
    has_ascii_output = bool(output_views)
    combined_views: list[str] = []
    for view in input_views + output_views:
        if view and view not in combined_views:
            combined_views.append(view)

    output_schema = _build_output_schema(
        ascii_views=output_views,
        ascii_format_output=has_ascii_output,
        include_answer=include_answer,
        include_description_output=include_description_output,
        ascii_order=ascii_order,
    )

    def _phrase(items: list[str]) -> str:
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"

    input_components: list[str] = []
    if include_query:
        input_components.append("a directional question")
    if task_inputs.get("description"):
        input_components.append("a scene description")
    if has_ascii_input:
        input_components.append("ASCII diagram(s)")
    if not input_components:
        input_components.append("the provided prompt")

    output_components: list[str] = []
    if include_answer:
        output_components.append("produce the correct directional answer")
    if has_ascii_output:
        output_components.append("generate the requested ASCII visualization")
    if include_description_output:
        output_components.append("write a concise spatial summary")

    base_intro_lines = [
        "Assume you have expertise in spatial reasoning. Your task is to answer the following question to the best of your knowledge.",
        f"You will receive {_phrase(input_components)}.",
    ]
    if output_components:
        base_intro_lines.append(
            f"Your job is to {_phrase(output_components)} and return the result as a compact JSON object."
        )
    else:
        base_intro_lines.append("Provide the required output as a compact JSON object.")

    template_sections: list[str] = []
    template_sections.append("\n".join(base_intro_lines))

    if include_answer:
        template_sections.append(
            f"Note that the possible directions are defined as {direction_list}, and there will be only one correct answer."
        )

    TEXT_INSTRUCTION = (
        "10 Distinct (Mutually Exclusive) Directions\n"
        "Pure Directions — perfectly aligned:\n"
        "- left: directly left, no vertical offset\n"
        "- right: directly right, no vertical offset\n"
        "- above: directly above, no horizontal offset\n"
        "- below: directly below, no vertical offset\n"
        "Diagonal Directions — both vertical and horizontal offset: upper-left, upper-right, lower-left, lower-right\n"
        "Horizontal Alignment ONLY: same level; Vertical Alignment ONLY: same column"
    )

    ASCII_INSTRUCTION = (
        "- Treat the canvas like a map: the first printed row is the top, and the last row is the bottom.\n"
        "- The first column is the left edge; the last column is the right edge.\n"
        "- Above = higher row; below = lower row (do not invert the vertical axis).\n"
        "- Left = smaller column index; right = larger column index (do not mirror horizontally)."
    )

    RENDERING_RULES = (
        "Rendering & Positioning Rules\n\n"
        "Use given uppercase letters (A, B, C, …); don’t invent new ones.\n"
        "Grid size is flexible—expand rows/cols as needed to fit all points.\n"
        "Place all letters; if two overlap, stack them (e.g., “AF”).\n\n"
        "Relations are first → second (e.g., “D to F” means D relative to F).\n\n"
        "Positioning:\n"
        "above/below → same column\n"
        "left/right → same level\n"
        "diagonals (upper/lower-left/right) → one cell diagonally.\n"
        "Keep perfect alignment for cardinal directions.\n"
        "Adjacent items occupy neighboring cells; no diagonal drift.\n"
        "Maintain uniform cell sizes and consistent borders.\n\n"
        "Keep row alignment when depicting left/right relations; do not shift labels vertically.\n"
        "Keep column alignment when depicting above/below relations; do not shift labels horizontally.\n"
        "For diagonal relations, ensure the two entities occupy different rows and different columns.\n"
        "Avoid inconsistent cell widths and do not introduce extra or missing letters."
    )

    def _build_example_output() -> str:
        if include_answer and has_ascii_output:
            return (
                "{\n"
                '  "answer": "<direction>",\n'
                '  "ascii": {\n'
                + ",\n".join(
                    f'    "{view}": "{ASCII_VIEW_DESCRIPTIONS[view]}"'
                    for view in output_views
                )
                + "\n  }\n}"
            )
        if include_answer and not has_ascii_output and not include_description_output:
            return '{\n  "answer": "<direction>"\n}'
        if include_description_output and not include_answer and not has_ascii_output:
            return '{\n  "description": "<concise spatial summary>"\n}'
        if include_description_output and has_ascii_output and not include_answer:
            return (
                "{\n"
                '  "description": "<concise spatial summary>",\n'
                '  "ascii": {\n'
                + ",\n".join(
                    f'    "{view}": "{ASCII_VIEW_DESCRIPTIONS[view]}"'
                    for view in output_views
                )
                + "\n  }\n}"
            )
        if has_ascii_output and not include_answer:
            return (
                "{\n"
                '  "ascii": {\n'
                + ",\n".join(
                    f'    "{view}": "{ASCII_VIEW_DESCRIPTIONS[view]}"'
                    for view in output_views
                )
                + "\n  }\n}"
            )
        if include_answer and include_description_output and has_ascii_output:
            return (
                "{\n"
                '  "answer": "<direction>",\n'
                '  "description": "<concise spatial summary>",\n'
                '  "ascii": {\n'
                + ",\n".join(
                    f'    "{view}": "{ASCII_VIEW_DESCRIPTIONS[view]}"'
                    for view in output_views
                )
                + "\n  }\n}"
            )
        if include_answer and include_description_output and not has_ascii_output:
            return (
                "{\n"
                '  "answer": "<direction>",\n'
                '  "description": "<concise spatial summary>"\n'
                "}"
            )
        schema_block = output_schema.replace("{", "{\n  ").replace("}", "\n}")
        if include_answer and has_ascii_output:
            return (
                "{\n"
                '  "answer": "<direction>",\n'
                '  "ascii": {\n'
                + ",\n".join(
                    f'    "{view}": "{ASCII_VIEW_DESCRIPTIONS[view]}"'
                    for view in output_views
                )
                + "\n  }\n}"
            )
        return schema_block

    if simple_mode:
        simple_lines: list[str] = [template_sections[0]]
        if include_answer:
            simple_lines.append(template_sections[1])
            simple_lines.append(TEXT_INSTRUCTION)
        if has_ascii_input or has_ascii_output:
            simple_lines.append(ASCII_INSTRUCTION)
            if combined_views:
                formats_block = "Formats:\n" + "\n".join(
                    f"- {view}: {ASCII_VIEW_DESCRIPTIONS[view]}"
                    for view in combined_views
                )
                simple_lines.append(formats_block)
        if has_ascii_output:
            simple_lines.append(RENDERING_RULES)
            extra_ascii = task_meta.get("ascii_guidelines", [])
            if extra_ascii:
                simple_lines.append("\n".join(extra_ascii))
            if output_views:
                view_list = ", ".join(output_views)
                simple_lines.append(f"Render the requested ASCII view(s): {view_list}.")
                descriptions = "\n".join(
                    f"- {view}: {ASCII_VIEW_DESCRIPTIONS[view]}"
                    for view in output_views
                )
                simple_lines.append("View Guidelines:\n" + descriptions)
            if include_answer:
                order_note = (
                    "Generate the ASCII visualization before finalizing the answer."
                    if ascii_order == "ascii_first"
                    else "Determine the correct answer first, then draw ASCII that matches it."
                )
                simple_lines.append(order_note)
        simple_lines.extend(task_meta.get("input_notes", []))
        simple_lines.extend(task_meta.get("output_notes", []))
        if include_answer:
            simple_lines.extend(task_meta.get("direction_notes", []))
        if include_description_output:
            simple_lines.extend(task_meta.get("description_guidelines", []))
        simple_lines.append(
            "Return ONLY a compact JSON object.\nDo not include markdown fences or any extra text."
        )
        return "\n\n".join(simple_lines)

    sections: list[str] = template_sections.copy()

    sections.append(f"{task_meta['title']}.\n{task_meta['objective']}")

    if include_answer:
        sections.append(TEXT_INSTRUCTION)
    if has_ascii_input or has_ascii_output:
        sections.append(ASCII_INSTRUCTION)
        if combined_views:
            sections.append(
                "Formats:\n"
                + "\n".join(
                    f"- {view}: {ASCII_VIEW_DESCRIPTIONS[view]}"
                    for view in combined_views
                )
            )
    if has_ascii_output:
        sections.append(RENDERING_RULES)
        extra_ascii = task_meta.get("ascii_guidelines", [])
        if extra_ascii:
            sections.append(
                "Additional ASCII Guidance:\n"
                + "\n".join(f"- {note}" for note in extra_ascii)
            )
        if include_answer:
            order_note = (
                "Generate the ASCII visualization before finalizing the answer."
                if ascii_order == "ascii_first"
                else "Determine the correct answer first, then draw ASCII that matches it."
            )
            sections.append(order_note)

    input_notes = task_meta.get("input_notes", [])
    output_notes = task_meta.get("output_notes", [])
    if input_notes:
        sections.append(
            "Input Notes:\n" + "\n".join(f"- {note}" for note in input_notes)
        )
    if output_notes:
        sections.append(
            "Output Notes:\n" + "\n".join(f"- {note}" for note in output_notes)
        )

    if include_answer:
        direction_notes = task_meta.get("direction_notes", [])
        if direction_notes:
            sections.append(
                "Direction Notes:\n"
                + "\n".join(f"- {note}" for note in direction_notes)
            )
    if include_description_output:
        description_notes = task_meta.get("description_guidelines", [])
        if description_notes:
            sections.append(
                "Description Guidelines:\n"
                + "\n".join(f"- {note}" for note in description_notes)
            )

    sections.append("OUTPUT SCHEMA (JSON):\n" + output_schema)

    sections.append(
        "Return ONLY a compact JSON object.\nDo not include markdown fences or any extra text."
    )

    return "\n\n".join(sections)


def build_messages(
    question,
    description="",
    few_shot_examples=None,
    ascii_format_input=None,
    ascii_format_output=None,
    prompt_mode="simple",
    few_shot_mode="system",
    ascii_order="answer_first",
    ascii_input=None,
    task="C",
):
    """
    Build the full messages array with flexible prompt configuration.

    Args:
        question: Task query (optional for task modes without queries).
        description: Scene description text.
        few_shot_examples: Optional list of example dicts with keys such as
            description, query_relation, label, and ascii.
        ascii_format_input: String/list of ASCII view names supplied as input evidence.
            - None / empty → no ASCII evidence expected unless task demands it.
            - str → single view (e.g., "grid").
            - list[str] → explicit set of views (subset of VALID_ASCII_VIEWS).
        ascii_format_output: String/list of ASCII view names to request in output.
            - None / empty → no ASCII output unless required by the task.
            - str → single view.
            - list[str] → explicit set of views.
        prompt_mode: "simple" or "detailed".
        few_shot_mode: "system", "user", or "conversational".
        ascii_order: "answer_first" or "ascii_first" (only relevant when both answer and ASCII are outputs).
        ascii_input: ASCII evidence for the current task (dict or str) when required.
        task: Task identifier ("A"–"F") controlling input/output expectations.

    Returns:
        list[dict]: Messages ready for the chat completion API.
    """

    task_config = TASK_CONFIGS.get(task, TASK_CONFIGS["C"])
    task_inputs = task_config["inputs"]
    task_outputs = task_config["outputs"]

    input_views = _normalize_ascii_views(ascii_format_input)
    output_views = _normalize_ascii_views(ascii_format_output)

    task_requires_ascii_input = task_inputs.get("ascii", False)
    task_requires_ascii_output = task_outputs.get("ascii", False)

    if not task_requires_ascii_input:
        input_views = []
    if not task_requires_ascii_output:
        output_views = []

    if task_requires_ascii_input and not input_views:
        if isinstance(ascii_input, dict):
            inferred = [
                view
                for view in VALID_ASCII_VIEWS
                if view in ascii_input and ascii_input[view]
            ]
            input_views = inferred or ["grid"]
        else:
            input_views = ["grid"]

    if task_requires_ascii_output and not output_views:
        output_views = ["grid"]

    expects_ascii_input = bool(input_views)
    has_ascii_output = bool(output_views)

    needs_answer = task_outputs.get("answer", False)
    needs_description_output = task_outputs.get("description", False)
    requires_query = task_inputs.get("query", False)
    requires_description_input = task_inputs.get("description", False)

    if requires_query and not question:
        raise ValueError(f"Task {task} expects a question/query but none was provided.")
    if requires_description_input and not description:
        raise ValueError(
            f"Task {task} expects a description input but none was provided."
        )
    if expects_ascii_input and ascii_input is None:
        raise ValueError(
            f"Task {task} expects ASCII input but ascii_input was not provided."
        )

    valid_prompt_modes = ["simple", "detailed"]
    valid_few_shot_modes = ["system", "user", "conversational"]
    valid_ascii_orders = ["answer_first", "ascii_first"]

    if prompt_mode not in valid_prompt_modes:
        raise ValueError(
            f"prompt_mode must be one of {valid_prompt_modes}, got '{prompt_mode}'"
        )
    if few_shot_mode not in valid_few_shot_modes:
        raise ValueError(
            f"few_shot_mode must be one of {valid_few_shot_modes}, got '{few_shot_mode}'"
        )
    if ascii_order not in valid_ascii_orders:
        raise ValueError(
            f"ascii_order must be one of {valid_ascii_orders}, got '{ascii_order}'"
        )

    simple_mode = prompt_mode == "simple"
    system_prompt = build_system_prompt(
        task_code=task,
        ascii_format_input=input_views,
        ascii_format_output=output_views,
        include_answer=needs_answer,
        include_description_output=needs_description_output,
        include_query=requires_query,
        simple_mode=simple_mode,
        ascii_order=ascii_order,
    )

    few_shot_in_system = few_shot_mode == "system"
    few_shot_in_user = few_shot_mode == "user"

    messages = [{"role": "system", "content": system_prompt}]

    def _build_input_block(desc_text, ascii_block, query_text, view_selection=None):
        sections: list[str] = []
        if task_inputs.get("description") and desc_text:
            sections.append(f"Description:\n{desc_text}")
        if task_inputs.get("ascii") and ascii_block:
            ascii_text = _format_ascii_block(ascii_block, views=view_selection)
            if ascii_text:
                sections.append(f"ASCII Reference:\n{ascii_text}")
        if task_inputs.get("query") and query_text:
            sections.append(f"Question: {query_text}")
        return "\n\n".join(sections)

    if few_shot_examples:
        if few_shot_in_system:
            system_examples = "\n\nEXAMPLES:\n"
            for i, ex in enumerate(few_shot_examples, 1):
                system_examples += f"\nExample {i}:\n"
                system_examples += (
                    format_few_shot_example(
                        ex,
                        task_inputs,
                        task_outputs,
                        output_views,
                        ascii_order,
                        input_views if input_views else None,
                    )
                    + "\n"
                )
            messages[0]["content"] += system_examples
        elif few_shot_in_user:
            user_content = ["Here are illustrative examples:"]
            for i, ex in enumerate(few_shot_examples, 1):
                user_content.append(f"\nExample {i}:")
                user_content.append(
                    format_few_shot_example(
                        ex,
                        task_inputs,
                        task_outputs,
                        output_views,
                        ascii_order,
                        input_views if input_views else None,
                    )
                )
            user_content.append("\nPlease solve the new task below.")
            messages.append({"role": "user", "content": "\n".join(user_content)})
        else:
            for ex in few_shot_examples:
                user_example = _build_input_block(
                    ex.get("description"),
                    ex.get("ascii"),
                    ex.get("query_relation"),
                    input_views if input_views else None,
                )
                if user_example:
                    messages.append({"role": "user", "content": user_example})

                output_example = format_few_shot_example(
                    ex,
                    task_inputs,
                    task_outputs,
                    output_views,
                    ascii_order,
                    input_views if input_views else None,
                )
                # Extract just the JSON from the formatted example for assistant turn
                output_json = output_example.split("Output:", 1)[-1].strip()
                messages.append({"role": "assistant", "content": output_json})

    if few_shot_in_system:
        # After embedding examples in system prompt, we still need the final user turn.
        pass
    elif few_shot_in_user:
        # The examples were appended in the same user message; fall through to append final task.
        pass

    final_input = _build_input_block(
        description, ascii_input, question, input_views if input_views else None
    )
    if not final_input:
        final_input = "Provide the required JSON response."

    if few_shot_in_user and len(messages) > 1 and messages[-1]["role"] == "user":
        messages[-1]["content"] += "\n\n" + final_input
    else:
        messages.append({"role": "user", "content": final_input})

    return messages


def ask_qwen3(
    client,
    question,
    description="",
    few_shot_examples=None,
    ascii_format_input=False,
    ascii_format_output=False,
    prompt_mode="detailed",
    few_shot_mode="system",
    ascii_order="answer_first",
    ascii_input=None,
    task="C",
):
    """
    Query Qwen3 model with spatial relationship question.

    Args:
        client: OpenAI client instance
        question: str - The spatial query (e.g., "What's the position of A to B?")
        description: str - Spatial scene description
        few_shot_examples: List[Dict] - Optional few-shot examples
        ascii_format_input: str | list[str] | None - ASCII input view(s) supplied (e.g., "grid" or ["grid","panel"])
        ascii_format_output: str | list[str] | None - ASCII output view(s) to request from the model
        prompt_mode: str - "simple" or "detailed" (default: "detailed")
        few_shot_mode: str - "system", "user", or "conversational" (default: "system")
        ascii_order: str - "answer_first" or "ascii_first" (controls output field order and guidance)
        ascii_input: str | dict | None - ASCII evidence for the current task when required
        task: str - Task identifier ("A"–"F") driving input/output expectations

    Returns:
        tuple: (messages, response_content)
    """
    messages = build_messages(
        question,
        description,
        few_shot_examples,
        ascii_format_input,
        ascii_format_output,
        prompt_mode,
        few_shot_mode,
        ascii_order,
        ascii_input=ascii_input,
        task=task,
    )

    completion = client.chat.completions.create(
        model="qwen3",
        temperature=0.0,
        messages=messages,
    )

    return messages, completion.choices[0].message.content


def ask_llama3(
    client,
    question,
    description="",
    few_shot_examples=None,
    ascii_format_input=False,
    ascii_format_output=False,
    prompt_mode="detailed",
    few_shot_mode="system",
    ascii_order="answer_first",
    ascii_input=None,
    task="C",
):
    """
    Query Llama3 model with spatial relationship question.

    Args:
        client: OpenAI client instance
        question: str - The spatial query (e.g., "What's the position of A to B?")
        description: str - Spatial scene description
        few_shot_examples: List[Dict] - Optional few-shot examples
        ascii_format_input: str | list[str] | None - ASCII input view(s) supplied (e.g., "grid" or ["grid","panel"])
        ascii_format_output: str | list[str] | None - ASCII output view(s) to request from the model
        prompt_mode: str - "simple" or "detailed" (default: "detailed")
        few_shot_mode: str - "system", "user", or "conversational" (default: "system")
        ascii_order: str - "answer_first" or "ascii_first" (controls output field order and guidance)
        ascii_input: str | dict | None - ASCII evidence for the current task when required
        task: str - Task identifier ("A"–"F") driving input/output expectations

    Returns:
        tuple: (messages, response_content)
    """
    messages = build_messages(
        question,
        description,
        few_shot_examples,
        ascii_format_input,
        ascii_format_output,
        prompt_mode,
        few_shot_mode,
        ascii_order,
        ascii_input=ascii_input,
        task=task,
    )

    completion = client.chat.completions.create(
        model="llama3-sdsc",
        temperature=0.0,
        messages=messages,
    )

    return messages, completion.choices[0].message.content


def ask_llama_hf_3b_instruct(
    pipe,
    question,
    description="",
    few_shot_examples=None,
    ascii_format_input=False,
    ascii_format_output=False,
    max_new_tokens=512,
    prompt_mode="simple",
    few_shot_mode="system",
    ascii_order="answer_first",
    ascii_input=None,
    task="C",
):
    """
    Query Llama 3.2 3B Instruct model via HuggingFace pipeline with spatial relationship question.

    Args:
        pipe: HuggingFace pipeline instance (text-generation with meta-llama/Llama-3.2-3B-Instruct)
        question: str - The spatial query (e.g., "What's the position of A to B?")
        description: str - Spatial scene description
        few_shot_examples: List[Dict] - Optional few-shot examples
        ascii_format_input: str | list[str] | None - ASCII input view(s) supplied (e.g., "grid" or ["grid","panel"])
        ascii_format_output: str | list[str] | None - ASCII output view(s) to request from the model
        max_new_tokens: int - Maximum number of tokens to generate (default 512)
        prompt_mode: str - "simple" or "detailed" (default: "simple" for 3B models)
        few_shot_mode: str - "system", "user", or "conversational" (default: "system" for 3B models)
        ascii_order: str - "answer_first" or "ascii_first" (controls output field order and guidance)
        ascii_input: str | dict | None - ASCII evidence for the current task when required
        task: str - Task identifier ("A"–"F") driving input/output expectations

    Returns:
        tuple: (messages, response_content)
    """
    messages = build_messages(
        question,
        description,
        few_shot_examples,
        ascii_format_input,
        ascii_format_output,
        prompt_mode,
        few_shot_mode,
        ascii_order,
        ascii_input=ascii_input,
        task=task,
    )

    generation_config = {"do_sample": False, "max_new_tokens": max_new_tokens}

    result = pipe(messages, **generation_config)
    return messages, result[0]["generated_text"][-1]["content"]
