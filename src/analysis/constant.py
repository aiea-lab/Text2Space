ANALYSIS_CONFIG = {
    "spatial": ["num_components"],
    "query": ["query_type"],
    "description": [
        "terminology_used",
        "has_unique_layout",
        "ambiguous_stages",
        "num_relations",
    ],
    "query+description": ["is_directly_stated"],
}

ANALYSIS_CONFIG_BY_TASK = {
    "A": ["spatial", "description"],
    "B": ["spatial"],
    "C": ["query+description", "query", "description"],
    "D": ["spatial", "query"],
    "E": ["query+description", "query", "description"],
    "E_ANSWER_FIRST": ["query+description", "query", "description"],
    "E_ASCII_FIRST": ["query+description", "query", "description"],
    "F": ["query+description", "query", "description", "spatial"],
    "G": ["query+description", "query", "description"],  # Same as E
    "H": ["query+description", "query", "description", "spatial"],  # Same as F
}
