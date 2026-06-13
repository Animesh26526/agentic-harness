import json
import os
import pytest
from harness.evaluators.rule_based import RuleBasedValidator
from harness.evaluators.semantic import SemanticEvaluator
from harness.evaluators.critic import CriticEvaluator
from harness.scoring import compute_reliability

def test_challenge_dataset_structure():
    """Verify that data/challenge_dataset.json exists and has the correct keys/schema."""
    dataset_path = "data/challenge_dataset.json"
    assert os.path.exists(dataset_path), "Challenge dataset file is missing!"
    
    with open(dataset_path, "r") as f:
        data = json.load(f)
        
    assert isinstance(data, list), "Challenge dataset must be a JSON list"
    assert len(data) == 14, "Challenge dataset must contain exactly 14 challenge samples"
    
    required_keys = {"query_id", "category", "description", "input", "expected_output", "evaluation_config"}
    for idx, sample in enumerate(data):
        assert required_keys.issubset(sample.keys()), f"Sample {idx} missing required keys"
        assert sample["category"] in {"structured_json", "constraint_following", "factual_qa"}, f"Sample {idx} has invalid category"
        assert isinstance(sample["evaluation_config"], dict), f"Sample {idx} evaluation_config must be a dict"


def test_challenge_dataset_rule_validator_evaluation():
    """Verify that deterministic rule validators evaluate the challenge dataset configurations without errors."""
    with open("data/challenge_dataset.json", "r") as f:
        data = json.load(f)
        
    validator = RuleBasedValidator()
    
    for sample in data:
        category = sample["category"]
        config = sample["evaluation_config"]
        expected = sample["expected_output"]
        
        # Test Category A and D deterministic limits and keywords
        if category in ("constraint_following", "structured_json"):
            result = validator.evaluate(
                generated_text=expected,
                validate_json=config.get("validate_json", False),
                required_fields=config.get("required_fields"),
                field_types=config.get("field_types"),
                forbidden_keywords=config.get("forbidden_keywords"),
                max_length=config.get("max_length"),
                min_length=config.get("min_length"),
                min_words=config.get("min_words"),
                max_words=config.get("max_words")
            )
            assert result.passed is True, f"Failed on query_id {sample['query_id']}: {result.issues}"
            assert result.score == 1.0
            assert isinstance(result.issues, list)


def test_challenge_dataset_scoring_integration():
    """Verify that compute_reliability correctly scores samples on the challenge dataset."""
    with open("data/challenge_dataset.json", "r") as f:
        data = json.load(f)
        
    for sample in data:
        category = sample["category"]
        
        # Verify that score is computed and matches weights
        if category == "structured_json":
            res = compute_reliability(category, rule_score=1.0)
            assert res["overall_score"] == 1.0
            assert res["objective_score"] == 1.0
            assert res["subjective_score"] == 1.0
        elif category == "constraint_following":
            res = compute_reliability(category, rule_score=1.0, critic_score=0.9)
            assert pytest.approx(res["overall_score"]) == 0.4 * 1.0 + 0.6 * 0.9
            assert res["objective_score"] == 1.0
            assert res["subjective_score"] == 0.9
        elif category == "factual_qa":
            res = compute_reliability(category, semantic_score=0.8, critic_score=0.9)
            assert pytest.approx(res["overall_score"]) == 0.4 * 0.8 + 0.6 * 0.9
            assert res["objective_score"] == 1.0
            assert pytest.approx(res["subjective_score"]) == 0.4 * 0.8 + 0.6 * 0.9
