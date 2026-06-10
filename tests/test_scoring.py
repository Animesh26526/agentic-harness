import pytest
from harness.scoring import compute_reliability
from harness.config import Config

def test_structured_json_scoring():
    """Verify structured_json scoring: R = rule_score."""
    res = compute_reliability("structured_json", rule_score=0.9)
    assert res["overall_score"] == 0.9
    assert res["passed"] is True
    assert res["threshold"] == Config.RELIABILITY_THRESHOLD

    # Borderline below threshold
    res2 = compute_reliability("structured_json", rule_score=0.79)
    assert res2["overall_score"] == 0.79
    assert res2["passed"] is False

def test_constraint_following_scoring():
    """Verify constraint_following scoring: R = 0.4 * rule_score + 0.6 * critic_score."""
    # 0.4 * 1.0 + 0.6 * 0.8 = 0.4 + 0.48 = 0.88 (passed)
    res = compute_reliability("constraint_following", rule_score=1.0, critic_score=0.8)
    assert pytest.approx(res["overall_score"]) == 0.88
    assert res["passed"] is True

    # 0.4 * 0.5 + 0.6 * 0.7 = 0.2 + 0.42 = 0.62 (failed)
    res2 = compute_reliability("constraint_following", rule_score=0.5, critic_score=0.7)
    assert pytest.approx(res2["overall_score"]) == 0.62
    assert res2["passed"] is False

def test_factual_qa_scoring():
    """Verify factual_qa scoring: R = 0.4 * semantic_score + 0.6 * critic_score."""
    # 0.4 * 0.9 + 0.6 * 0.9 = 0.36 + 0.54 = 0.90 (passed)
    res = compute_reliability("factual_qa", semantic_score=0.9, critic_score=0.9)
    assert pytest.approx(res["overall_score"]) == 0.90
    assert res["passed"] is True

    # 0.4 * 0.7 + 0.6 * 0.8 = 0.28 + 0.48 = 0.76 (failed)
    res2 = compute_reliability("factual_qa", semantic_score=0.7, critic_score=0.8)
    assert pytest.approx(res2["overall_score"]) == 0.76
    assert res2["passed"] is False

def test_extraction_math_scoring():
    """Verify extraction_math scoring: R = 0.5 * semantic_score + 0.5 * rule_score."""
    # 0.5 * 0.8 + 0.5 * 0.9 = 0.40 + 0.45 = 0.85 (passed)
    res = compute_reliability("extraction_math", semantic_score=0.8, rule_score=0.9)
    assert pytest.approx(res["overall_score"]) == 0.85
    assert res["passed"] is True

    # 0.5 * 0.6 + 0.5 * 0.8 = 0.30 + 0.40 = 0.70 (failed)
    res2 = compute_reliability("extraction_math", semantic_score=0.6, rule_score=0.8)
    assert pytest.approx(res2["overall_score"]) == 0.70
    assert res2["passed"] is False

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
