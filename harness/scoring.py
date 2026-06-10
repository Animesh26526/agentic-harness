from typing import Any, Dict, Optional
from harness.config import Config

def compute_reliability(
    category: str,
    semantic_score: Optional[float] = None,
    rule_score: Optional[float] = None,
    critic_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes overall reliability score based on task category and individual evaluator scores.

    Args:
        category (str): The task category. Must be one of structured_json, constraint_following,
                        factual_qa, or extraction_math.
        semantic_score (float, optional): Semantic similarity score.
        rule_score (float, optional): Rule-based validation score.
        critic_score (float, optional): Critic judge score.

    Returns:
        Dict[str, Any]: A dictionary containing overall_score (float), passed (bool), and threshold (float).

    Raises:
        ValueError: If category is invalid, or if any required score for the category is missing.
    """
    # 1. Validate Category
    valid_categories = {"structured_json", "constraint_following", "factual_qa", "extraction_math"}
    if category not in valid_categories:
        raise ValueError(
            f"Invalid category: '{category}'. Expected one of: {list(valid_categories)}"
        )

    # Helper function to ensure score is float
    def clean_score(val: Any, name: str) -> float:
        if val is None:
            raise ValueError(f"Required score '{name}' is missing for category '{category}'")
        try:
            return float(val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Score '{name}' must be a number, got: {val}") from e

    # 2. Validation & Score calculation per category
    if category == "structured_json":
        r_score = clean_score(rule_score, "rule_score")
        overall_score = r_score

    elif category == "constraint_following":
        r_score = clean_score(rule_score, "rule_score")
        c_score = clean_score(critic_score, "critic_score")
        overall_score = 0.4 * r_score + 0.6 * c_score

    elif category == "factual_qa":
        s_score = clean_score(semantic_score, "semantic_score")
        c_score = clean_score(critic_score, "critic_score")
        overall_score = 0.4 * s_score + 0.6 * c_score

    elif category == "extraction_math":
        s_score = clean_score(semantic_score, "semantic_score")
        r_score = clean_score(rule_score, "rule_score")
        overall_score = 0.5 * s_score + 0.5 * r_score

    # Clamp overall score to [0.0, 1.0]
    overall_score = max(0.0, min(1.0, overall_score))

    threshold = Config.RELIABILITY_THRESHOLD
    passed = overall_score >= threshold

    return {
        "overall_score": overall_score,
        "passed": passed,
        "threshold": threshold
    }
