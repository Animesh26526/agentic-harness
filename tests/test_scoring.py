import pytest
from harness.scoring import compute_reliability
from harness.config import Config

def test_structured_json_scoring():
    """Verify structured_json scoring: R = rule_score."""
    res = compute_reliability("structured_json", rule_score=1.0)
    assert res["overall_score"] == 1.0
    assert res["passed"] is True
    assert res["threshold"] == Config.RELIABILITY_THRESHOLD

    # Borderline below threshold (capped at 0.49)
    res2 = compute_reliability("structured_json", rule_score=0.79)
    assert res2["overall_score"] == 0.49
    assert res2["passed"] is False

    # Score above threshold but rule_score < 1.0 (fails objective check, capped at 0.49)
    res3 = compute_reliability("structured_json", rule_score=0.9)
    assert res3["overall_score"] == 0.49
    assert res3["passed"] is False

def test_constraint_following_scoring():
    """Verify constraint_following scoring: R = 0.4 * rule_score + 0.6 * critic_score."""
    # 0.4 * 1.0 + 0.6 * 0.8 = 0.4 + 0.48 = 0.88 (passed)
    res = compute_reliability("constraint_following", rule_score=1.0, critic_score=0.8)
    assert pytest.approx(res["overall_score"]) == 0.88
    assert res["passed"] is True

    # 0.4 * 0.5 + 0.6 * 0.7 = 0.2 + 0.42 = 0.62 (failed, capped at 0.49)
    res2 = compute_reliability("constraint_following", rule_score=0.5, critic_score=0.7)
    assert pytest.approx(res2["overall_score"]) == 0.49
    assert res2["passed"] is False

def test_factual_qa_scoring():
    """Verify factual_qa scoring: R = 0.4 * semantic_score + 0.6 * critic_score."""
    # 0.4 * 0.9 + 0.6 * 0.9 = 0.36 + 0.54 = 0.90 (passed)
    res = compute_reliability("factual_qa", semantic_score=0.9, critic_score=0.9)
    assert pytest.approx(res["overall_score"]) == 0.90
    assert res["passed"] is True

    # 0.4 * 0.7 + 0.6 * 0.8 = 0.28 + 0.48 = 0.76 (failed)
    # Note: Since the critic_score (0.8) is below the threshold, the overall score is capped to 0.49
    res2 = compute_reliability("factual_qa", semantic_score=0.7, critic_score=0.8)
    assert pytest.approx(res2["overall_score"]) == 0.49
    assert res2["passed"] is False

def test_extraction_math_scoring():
    """Verify extraction_math scoring: R = 0.5 * semantic_score + 0.5 * rule_score."""
    # 0.5 * 0.8 + 0.5 * 1.0 = 0.40 + 0.50 = 0.90 (passed)
    res = compute_reliability("extraction_math", semantic_score=0.8, rule_score=1.0)
    assert pytest.approx(res["overall_score"]) == 0.90
    assert res["passed"] is True

    # 0.5 * 0.6 + 0.5 * 0.8 = 0.30 + 0.40 = 0.70 (failed, capped at 0.49)
    res2 = compute_reliability("extraction_math", semantic_score=0.6, rule_score=0.8)
    assert pytest.approx(res2["overall_score"]) == 0.49
    assert res2["passed"] is False

    # Score above threshold but rule_score < 1.0 (fails objective check, capped at 0.49)
    res3 = compute_reliability("extraction_math", semantic_score=0.8, rule_score=0.9)
    assert pytest.approx(res3["overall_score"]) == 0.49
    assert res3["passed"] is False

def test_missing_scores_exceptions():
    """Verify that missing scores raise ValueError for appropriate categories."""
    # structured_json requires rule_score
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("structured_json")
    assert "rule_score" in str(excinfo.value)

    # constraint_following requires rule_score and critic_score
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("constraint_following", rule_score=0.8)
    assert "critic_score" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("constraint_following", critic_score=0.8)
    assert "rule_score" in str(excinfo.value)

    # factual_qa requires semantic_score and critic_score
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("factual_qa", semantic_score=0.8)
    assert "critic_score" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("factual_qa", critic_score=0.8)
    assert "semantic_score" in str(excinfo.value)

    # extraction_math requires semantic_score and rule_score
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("extraction_math", semantic_score=0.8)
    assert "rule_score" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        compute_reliability("extraction_math", rule_score=0.8)
    assert "semantic_score" in str(excinfo.value)

def test_invalid_category():
    """Verify that invalid category raises ValueError."""
    with pytest.raises(ValueError):
        compute_reliability("unknown_category", rule_score=0.9)


def test_layered_scoring_outputs():
    """Verify objective_score and subjective_score outputs from compute_reliability."""
    # 1. structured_json: w_obj = 1.0, w_subj = 0.0
    res = compute_reliability("structured_json", rule_score=0.85)
    assert res["objective_score"] == 0.85
    assert res["subjective_score"] == 1.0  # default
    assert res["weights"]["objective"] == 1.0
    assert res["weights"]["subjective"] == 0.0

    # 2. constraint_following: w_obj = 0.4, w_subj = 0.6
    res2 = compute_reliability("constraint_following", rule_score=0.80, critic_score=0.90)
    assert res2["objective_score"] == 0.80
    assert res2["subjective_score"] == 0.90
    assert res2["weights"]["objective"] == 0.4
    assert res2["weights"]["subjective"] == 0.6

    # 3. factual_qa: w_obj = 0.0, w_subj = 1.0
    res3 = compute_reliability("factual_qa", semantic_score=0.70, critic_score=0.80)
    assert res3["objective_score"] == 1.0  # default
    assert pytest.approx(res3["subjective_score"]) == 0.4 * 0.70 + 0.6 * 0.80
    assert res3["weights"]["objective"] == 0.0
    assert res3["weights"]["subjective"] == 1.0


def test_explanation_engine_markdown_generation():
    """Verify generate_reliability_explanation_markdown correctly checks rules and outputs formatted HTML."""
    from harness.scoring import generate_reliability_explanation_markdown

    # Case 1: All passed
    html = generate_reliability_explanation_markdown(
        overall_score=0.95,
        passed=True,
        category="constraint_following",
        rule_score=1.0,
        semantic_score=None,
        critic_score=0.95,
        issues=[],
        retry_count=0,
        evaluation_config={"max_length": 100, "forbidden_keywords": ["forbidden"]}
    )

    assert "Reliability: 95 / 100" in html
    assert "✓ Character Limit Passed" in html
    assert "✓ Keyword Constraints Passed" in html
    assert "✓ Critic Approved" in html
    assert "✓ No Retry Needed" in html

    # Case 2: Some failed
    html2 = generate_reliability_explanation_markdown(
        overall_score=0.55,
        passed=False,
        category="constraint_following",
        rule_score=0.4,
        semantic_score=None,
        critic_score=0.6,
        issues=["Response length is 120, exceeds limit of 100", "Forbidden word detected", "[Medium] Critic rejected"],
        retry_count=2,
        evaluation_config={"max_length": 100, "forbidden_keywords": ["forbidden"]}
    )

    assert "Reliability: 55 / 100" in html2
    assert "✗ Character Limit Violated" in html2
    assert "✗ Constraint Violation" in html2
    assert "✗ Critic Rejected" in html2
    assert "✗ Failed After 2 Retries" in html2
