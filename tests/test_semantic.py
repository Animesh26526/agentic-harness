import pytest
from harness.evaluators.semantic import SemanticEvaluator

def test_identical_text():
    """Verifies that identical sentences yield a near 1.0 similarity score."""
    evaluator = SemanticEvaluator(threshold=0.85)
    text = "The quick brown fox jumps over the lazy dog."
    result = evaluator.evaluate(text, text)
    
    assert result.score >= 0.99
    assert result.passed is True
    assert len(result.issues) == 0

def test_similar_text():
    """Verifies that semantically similar sentences pass the default similarity thresholds."""
    evaluator = SemanticEvaluator(threshold=0.70)
    gen_text = "A speedy brown fox leaps across a sleepy dog."
    ref_text = "The quick brown fox jumps over the lazy dog."
    result = evaluator.evaluate(gen_text, ref_text)
    
    assert result.score >= 0.70
    assert result.passed is True
    assert len(result.issues) == 0

def test_unrelated_text():
    """Verifies that completely unrelated topics fail the similarity check."""
    evaluator = SemanticEvaluator(threshold=0.60)
    gen_text = "Deep learning operates by training neural networks on massive datasets."
    ref_text = "The quick brown fox jumps over the lazy dog."
    result = evaluator.evaluate(gen_text, ref_text)
    
    assert result.score < 0.60
    assert result.passed is False
    assert len(result.issues) > 0
    assert "falls below threshold" in result.issues[0]

def test_empty_generated_text():
    """Confirms that empty generated inputs are caught gracefully, scoring 0.0 with fail status."""
    evaluator = SemanticEvaluator()
    
    # Empty generated response
    result_empty_gen = evaluator.evaluate("", "Reference text content.")
    assert result_empty_gen.passed is False
    assert result_empty_gen.score == 0.0
    assert "Missing generated_text or reference_text" in result_empty_gen.issues[0]

    # Empty reference content
    result_empty_ref = evaluator.evaluate("Generated text content.", "")
    assert result_empty_ref.passed is False
    assert result_empty_ref.score == 0.0
    assert "Missing generated_text or reference_text" in result_empty_ref.issues[0]
