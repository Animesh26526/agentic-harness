from typing import List

# Routing table mapping task categories to their designated evaluators
CATEGORY_TO_EVALUATORS = {
    "structured_json": [
        "rule"
    ],

    "constraint_following": [
        "rule",
        "critic"
    ],

    "factual_qa": [
        "semantic",
        "critic"
    ],

    "extraction_math": [
        "semantic",
        "rule"
    ]
}

def get_evaluators(category: str) -> List[str]:
    """
    Determines which evaluators should run for the given task category.

    Args:
        category (str): The task category to look up.

    Returns:
        List[str]: The list of evaluator names assigned to this category.

    Raises:
        ValueError: If the category is not registered or invalid.
    """
    if category not in CATEGORY_TO_EVALUATORS:
        raise ValueError(
            f"Invalid category: '{category}'. Expected one of: "
            f"{list(CATEGORY_TO_EVALUATORS.keys())}"
        )
    # Return a copy to prevent external mutation
    return list(CATEGORY_TO_EVALUATORS[category])
