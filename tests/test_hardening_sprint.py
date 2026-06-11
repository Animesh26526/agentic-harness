import pytest
from harness.evaluators.critic import CriticEvaluator, is_contradictory_violation
from harness.scoring import compute_reliability
from harness.orchestrator import Orchestrator

def test_task1_paraphrases_and_stronger_answers(monkeypatch):
    critic = CriticEvaluator()
    # Mock LLM to return a high score since the instructions now say to accept paraphrases
    monkeypatch.setattr(critic.agent, 'generate', lambda x: '{"score": 1.0, "issues": [], "suggestions": []}')
    res = critic.evaluate(
        generated_text="Colorless, odorless, tasteless, and transparent substance essential for life.",
        reference_text="Water is a transparent fluid.",
        user_query="What is water?"
    )
    assert res.passed is True
    assert res.score == 1.0

def test_task2_contradiction_detection():
    # 1. Existing keyword
    assert is_contradictory_violation("Response does not state explicitly the word transparent", "It is transparent.") == True
    # 2. Existing concept (semantic fallback)
    assert is_contradictory_violation("Missing transparent fluid", "It is a colorless and see-through substance.") == True

def test_task4_reliability_capping():
    # Passed run
    res = compute_reliability("factual_qa", semantic_score=0.9, rule_score=1.0, critic_score=0.9)
    assert res["passed"] is True
    assert res["overall_score"] >= 0.8
    
    # Failed run (critic rejected)
    res2 = compute_reliability("factual_qa", semantic_score=0.9, rule_score=1.0, critic_score=0.7)
    assert res2["passed"] is False
    assert res2["overall_score"] <= 0.49
    
    # Objective failed
    res3 = compute_reliability("structured_json", semantic_score=1.0, rule_score=0.8, critic_score=1.0)
    assert res3["passed"] is False
    assert res3["overall_score"] <= 0.49
