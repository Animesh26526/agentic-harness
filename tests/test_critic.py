import pytest
from unittest.mock import MagicMock, patch
from harness.evaluators.critic import CriticEvaluator
from harness.evaluators.base_evaluator import EvaluationResult

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_valid_json_pass(mock_agent_class):
    """Test valid JSON response from Gemini that passes the threshold."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = '{"score": 0.85, "issues": ["Missing age field"], "suggestions": ["Include the age field."]}'

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate(
        generated_text="name: Alice",
        reference_text="name: Alice, age: 30",
        user_query="Format the details"
    )

    assert isinstance(result, EvaluationResult)
    assert result.score == 0.85
    assert result.passed is True
    assert "Missing age field" in result.issues
    assert "Include the age field." in result.metadata["suggestions"]
    mock_agent.generate.assert_called_once()

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_valid_json_fail(mock_agent_class):
    """Test valid JSON response from Gemini that falls below the threshold."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = '{"score": 0.70, "issues": ["Unsupported claims"], "suggestions": ["Remove claims."]}'

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate(
        generated_text="name: Alice, age: 30, city: Paris",
        reference_text="name: Alice, age: 30",
        user_query="Format the details"
    )

    assert result.score == 0.70
    assert result.passed is False
    assert "Unsupported claims" in result.issues
    assert "Remove claims." in result.metadata["suggestions"]

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_invalid_json_unsalvageable(mock_agent_class):
    """Test completely malformed JSON response that cannot be recovered."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = "This response is not JSON at all."

    evaluator = CriticEvaluator()
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.0
    assert result.passed is False
    assert "Critic returned invalid JSON" in result.issues
    assert result.metadata["raw_response"] == "This response is not JSON at all."

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_invalid_json_markdown_recovery(mock_agent_class):
    """Test malformed JSON wrapped in markdown blocks is successfully recovered."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = """```json
    {
      "score": 0.90,
      "issues": [],
      "suggestions": []
    }
    ```"""

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.90
    assert result.passed is True
    assert len(result.issues) == 0

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_invalid_json_bracket_recovery(mock_agent_class):
    """Test JSON response embedded inside conversational text is recovered."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = """Sure! Here is the evaluation:
    {
      "score": 0.95,
      "issues": ["Minor typo"],
      "suggestions": ["Fix spelling"]
    }
    Hope this helps!"""

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.95
    assert result.passed is True
    assert "Minor typo" in result.issues

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_invalid_json_trailing_comma_recovery(mock_agent_class):
    """Test JSON response with trailing commas is recovered."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = '{"score": 0.80, "issues": ["error",], "suggestions": [],}'

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.80
    assert result.passed is True
    assert "error" in result.issues

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_invalid_json_single_quote_recovery(mock_agent_class):
    """Test JSON response using single quotes is recovered."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = "{'score': 0.80, 'issues': [], 'suggestions': []}"

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.80
    assert result.passed is True
    assert len(result.issues) == 0

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_missing_fields(mock_agent_class):
    """Test JSON response missing required fields is handled gracefully without crashing."""
    mock_agent = mock_agent_class.return_value
    # Missing 'issues' and 'suggestions'
    mock_agent.generate.return_value = '{"score": 0.85}'

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.85
    # Since there are validation errors, passed should be False
    assert result.passed is False
    assert any("missing required" in issue for issue in result.issues)

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_empty_or_whitespace_generated_text(mock_agent_class):
    """Test that empty or whitespace generated text fails immediately without calling the agent."""
    mock_agent = mock_agent_class.return_value

    evaluator = CriticEvaluator()
    result = evaluator.evaluate(
        generated_text="   ",
        reference_text="ref",
        user_query="query"
    )

    assert result.score == 0.0
    assert result.passed is False
    assert "Generated text is empty." in result.issues
    mock_agent.generate.assert_not_called()

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_agent_exception(mock_agent_class):
    """Test that agent exceptions are caught gracefully and score is 0.0."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.side_effect = Exception("API connection timed out")

    evaluator = CriticEvaluator()
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.0
    assert result.passed is False
    assert any("Gemini API call failed during evaluation" in issue for issue in result.issues)
