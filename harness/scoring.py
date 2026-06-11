from typing import Any, Dict, Optional
from harness.config import Config

from typing import Any, Dict, List, Optional
from harness.config import Config

def compute_reliability(
    category: str,
    semantic_score: Optional[float] = None,
    rule_score: Optional[float] = None,
    critic_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes overall reliability score based on task category and individual evaluator scores
    by separating the assessment into explicit Objective and Subjective evaluation layers.

    Scoring Architecture Formula:
    ------------------------------
    Overall Score = w_obj * Objective_Score + w_subj * Subjective_Score

    Where:
    - Objective_Score is derived from deterministic rule-based checks (RuleBasedValidator)
      representing character limits, JSON syntax, required fields, and forbidden keywords.
      Defaults to 1.0 if no objective rules apply.
    - Subjective_Score is derived from semantic similarity (SemanticEvaluator) and/or
      LLM-as-a-judge critique (CriticEvaluator), representing coherence, helpfulness,
      reasoning quality, and semantic accuracy. Defaults to 1.0 if no subjective evaluators apply.

    Weights (w_obj, w_subj) by category:
    - structured_json: w_obj = 1.0, w_subj = 0.0  (Strict format compliance)
    - constraint_following: w_obj = 0.4, w_subj = 0.6  (Hybrid format & quality checks)
    - factual_qa: w_obj = 0.0, w_subj = 1.0  (Meaning-based check against ground truth)
    - extraction_math: w_obj = 0.5, w_subj = 0.5  (Hybrid exact match and similarity check)

    Args:
        category (str): The task category. Must be one of structured_json, constraint_following,
                        factual_qa, or extraction_math.
        semantic_score (float, optional): Semantic similarity score.
        rule_score (float, optional): Rule-based validation score.
        critic_score (float, optional): Critic judge score.

    Returns:
        Dict[str, Any]: A dictionary containing overall_score (float), passed (bool), threshold (float),
                        objective_score (float), subjective_score (float), and weights (dict).

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

    # 2. Strict category-specific required score validation and layer mapping
    if category == "structured_json":
        r_score = clean_score(rule_score, "rule_score")
        objective_score = r_score
        subjective_score = 1.0
    elif category == "constraint_following":
        r_score = clean_score(rule_score, "rule_score")
        c_score = clean_score(critic_score, "critic_score")
        objective_score = r_score
        subjective_score = c_score
    elif category == "factual_qa":
        s_score = clean_score(semantic_score, "semantic_score")
        c_score = clean_score(critic_score, "critic_score")
        objective_score = 1.0
        subjective_score = 0.4 * s_score + 0.6 * c_score
    elif category == "extraction_math":
        s_score = clean_score(semantic_score, "semantic_score")
        r_score = clean_score(rule_score, "rule_score")
        objective_score = r_score
        subjective_score = s_score
    else:
        objective_score = 1.0
        subjective_score = 1.0

    # 3. Transparent Weighted Combination per category
    if category == "structured_json":
        w_obj, w_subj = 1.0, 0.0
    elif category == "constraint_following":
        w_obj, w_subj = 0.4, 0.6
    elif category == "factual_qa":
        w_obj, w_subj = 0.0, 1.0
    elif category == "extraction_math":
        w_obj, w_subj = 0.5, 0.5
    else:
        w_obj, w_subj = 0.5, 0.5

    overall_score = w_obj * objective_score + w_subj * subjective_score

    # Clamp overall score to [0.0, 1.0]
    overall_score = max(0.0, min(1.0, overall_score))

    # Cap reliability score below acceptance threshold if any objective check fails
    if objective_score < 1.0:
        overall_score = min(overall_score, 0.49)

    threshold = Config.RELIABILITY_THRESHOLD
    passed = (overall_score >= threshold) and (objective_score >= 1.0)

    return {
        "overall_score": overall_score,
        "passed": passed,
        "threshold": threshold,
        "objective_score": objective_score,
        "subjective_score": subjective_score,
        "weights": {"objective": w_obj, "subjective": w_subj}
    }


def generate_reliability_explanation_markdown(
    overall_score: float,
    passed: bool,
    category: str,
    rule_score: Optional[float],
    semantic_score: Optional[float],
    critic_score: Optional[float],
    issues: List[str],
    retry_count: int,
    evaluation_config: Dict[str, Any]
) -> str:
    """
    Generates a human-readable, checklist-style markdown explanation block for the overall reliability score.

    Args:
        overall_score (float): Composite reliability score.
        passed (bool): Verdict.
        category (str): Evaluated task category.
        rule_score (float, optional): Objective validation score.
        semantic_score (float, optional): Semantic score.
        critic_score (float, optional): LLM critic score.
        issues (List[str]): List of detected validation issues/violations.
        retry_count (int): Number of retries triggered during self-correction.
        evaluation_config (dict): Target evaluation parameters.

    Returns:
        str: Styled HTML block representing the explanation.
    """
    percentage = int(round(overall_score * 100))
    checklist = []

    # 1. Character Limit
    max_length = evaluation_config.get("max_length")
    if max_length is not None:
        has_length_issue = any("length" in iss.lower() or "limit" in iss.lower() for iss in issues)
        if has_length_issue:
            checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Character Limit Violated</span>")
        else:
            checklist.append("<span style='color: #10b981; font-weight: bold;'>✓ Character Limit Passed</span>")

    # 2. Keyword Constraints
    forbidden_kws = evaluation_config.get("forbidden_keywords")
    if forbidden_kws:
        has_kw_issue = any("forbidden" in iss.lower() or "keyword" in iss.lower() for iss in issues)
        if has_kw_issue:
            checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Constraint Violation</span>")
        else:
            checklist.append("<span style='color: #10b981; font-weight: bold;'>✓ Keyword Constraints Passed</span>")

    # 3. JSON Validity & Schema
    validate_json = evaluation_config.get("validate_json", False)
    if validate_json:
        has_json_issue = any("json" in iss.lower() or "schema" in iss.lower() or "field" in iss.lower() for iss in issues)
        if has_json_issue:
            is_missing = any("missing required" in iss.lower() or "schema violation" in iss.lower() for iss in issues)
            if is_missing:
                checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Missing JSON Field</span>")
            else:
                checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Invalid JSON Format</span>")
        else:
            checklist.append("<span style='color: #10b981; font-weight: bold;'>✓ JSON Schema Passed</span>")

    # 4. Semantic Similarity
    if category in ("factual_qa", "extraction_math") and semantic_score is not None:
        has_semantic_issue = any("semantic" in iss.lower() or "similarity" in iss.lower() for iss in issues)
        if has_semantic_issue:
            checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Semantic Similarity Low</span>")
        else:
            checklist.append("<span style='color: #10b981; font-weight: bold;'>✓ Semantic Similarity Acceptable</span>")

    # 5. Critic Approval
    if category in ("constraint_following", "factual_qa") and critic_score is not None:
        try:
            c_val = float(critic_score)
        except (TypeError, ValueError):
            c_val = 0.0
        if c_val < Config.CRITIC_THRESHOLD:
            checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Critic Rejected</span>")
        else:
            checklist.append("<span style='color: #10b981; font-weight: bold;'>✓ Critic Approved</span>")

    # 6. Retry Status
    if retry_count == 0:
        if passed:
            checklist.append("<span style='color: #10b981; font-weight: bold;'>✓ No Retry Needed</span>")
        else:
            checklist.append("<span style='color: #ef4444; font-weight: bold;'>✗ Retry Required</span>")
    else:
        label = "Retry" if retry_count == 1 else "Retries"
        if passed:
            checklist.append(f"<span style='color: #10b981; font-weight: bold;'>✓ Passed After {retry_count} {label}</span>")
        else:
            checklist.append(f"<span style='color: #ef4444; font-weight: bold;'>✗ Failed After {retry_count} {label}</span>")

    # Construct the final HTML block
    checklist_str = " &nbsp;•&nbsp; ".join(checklist)

    # Color coordinate card outline and background based on passing state
    border_color = "#10b981" if passed else "#ef4444"
    bg_color = "rgba(16, 185, 129, 0.08)" if passed else "rgba(239, 68, 68, 0.08)"

    html = f"""
    <div style="background-color: {bg_color}; border-radius: 12px; padding: 18px; border-left: 6px solid {border_color}; margin-top: 15px; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
        <div style="font-size: 1.25rem; font-weight: bold; color: #ffffff; margin-bottom: 8px;">
            Reliability: {percentage} / 100
        </div>
        <div style="font-size: 0.95rem; color: #cbd5e1; line-height: 1.6;">
            <strong style="color: #f8fafc;">Reason Checklist:</strong><br/>
            {checklist_str}
        </div>
    </div>
    """
    return html
