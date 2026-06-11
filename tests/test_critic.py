import json
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

@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_multiple_markdown_blocks_recovery(mock_agent_class):
    """Test that valid JSON block is recovered when preceded by a non-JSON code block (e.g. python)."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = """Here is some explanation code:
    ```python
    def dummy():
        return "not JSON"
    ```
    And here is the JSON:
    ```json
    {
      "score": 0.92,
      "issues": ["Test issue"],
      "suggestions": []
    }
    ```"""

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    assert result.score == 0.92
    assert result.passed is True
    assert "Test issue" in result.issues

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


@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_severity_parsing_and_filtering(mock_agent_class):
    """Test that Low/Informational severity issues are filtered out from violations but stored in metadata, and Critical/Medium are kept."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = json.dumps({
        "score": 0.75,
        "issues": [
            {"description": "The response has a critical hallucination", "severity": "Critical"},
            {"description": "Missing optional details about history", "severity": "Low"},
            {"description": "Minor phrasing feedback", "severity": "Informational"},
            {"description": "Response has medium quality issues", "severity": "Medium"}
        ],
        "suggestions": ["Fix hallucination"]
    })

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("gen", "ref", "query")

    # Only Critical and Medium issues should be in result.issues
    assert len(result.issues) == 2
    assert "The response has a critical hallucination" in result.issues
    assert "Response has medium quality issues" in result.issues

    # Low and Informational issues should be in metadata["low_info_issues"]
    low_info = result.metadata["low_info_issues"]
    assert len(low_info) == 2
    descriptions = [x["description"] for x in low_info]
    assert "Missing optional details about history" in descriptions
    assert "Minor phrasing feedback" in descriptions

    # Score was 0.75, threshold is 0.80, so passed is False
    assert result.score == 0.75
    assert result.passed is False


@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_calibration_to_one_when_only_low_issues(mock_agent_class):
    """Test that if the critic returns only Low/Informational issues, the score calibrates to 1.0 (passed)."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = json.dumps({
        "score": 0.80,
        "issues": [
            {"description": "Missing optional dessert info", "severity": "Low"},
            {"description": "Nice explanation overall", "severity": "Informational"}
        ],
        "suggestions": []
    })

    evaluator = CriticEvaluator(threshold=0.85)
    result = evaluator.evaluate("gen", "ref", "query")

    # All issues should be filtered out
    assert len(result.issues) == 0
    # Score must calibrate to 1.0
    assert result.score == 1.0
    assert result.passed is True


@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_contradiction_detection(mock_agent_class):
    """Test that a violation claiming the response does not state X is discarded when response explicitly contains X."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = json.dumps({
        "score": 0.0,
        "issues": [
            {"description": "The response failed to explicitly state that the query's concept is not valid.", "severity": "Critical", "confidence": "High"},
            {"description": "The response is missing the explanation of quantum gates", "severity": "Medium", "confidence": "High"}
        ],
        "suggestions": ["Add gates."]
    })

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate(
        generated_text="There is no such concept, and it is not valid.",
        reference_text="Not valid.",
        user_query="Query"
    )

    assert len(result.issues) == 1
    assert "missing the explanation" in result.issues[0]
    assert result.score == 0.85
    assert result.passed is True


@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_evaluator_confidence_demotion(mock_agent_class):
    """Test that low-confidence findings are demoted to Low severity and thus do not trigger retries or score penalties."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = json.dumps({
        "score": 0.40,
        "issues": [
            {"description": "Borderline quality issue", "severity": "Critical", "confidence": "Low"},
            {"description": "Another minor issue", "severity": "Medium", "confidence": "High"}
        ],
        "suggestions": []
    })

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate("some text", "some ref", "query")

    assert len(result.issues) == 1
    assert "Another minor issue" in result.issues[0]

    low_info = result.metadata["low_info_issues"]
    assert len(low_info) == 1
    assert "Borderline quality issue" in low_info[0]["description"]

    assert result.score == 0.85
    assert result.passed is True


@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_hallucination_override(mock_agent_class):
    """Test that for hallucination tests, if the response contains objective success criteria, any critic violation stating it failed is discarded."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = json.dumps({
        "score": 0.0,
        "issues": [
            {"description": "The response did not state that the flibbertigibbet engine is not a recognized concept.", "severity": "Critical", "confidence": "High"},
            {"description": "The response lacks a title.", "severity": "Medium", "confidence": "High"}
        ],
        "suggestions": []
    })

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate(
        generated_text="The flibbertigibbet engine is a fictional and made-up concept.",
        reference_text="There is no such thing as a flibbertigibbet engine in quantum computing. It is a completely made-up concept.",
        user_query="Explain the functions of the flibbertigibbet engine in quantum computing."
    )

    assert len(result.issues) == 1
    assert "lacks a title" in result.issues[0]
    assert result.score == 0.85
    assert result.passed is True


@patch('harness.evaluators.critic.GeminiAgent')
def test_critic_fifa_hallucination_override(mock_agent_class):
    """Test that for the 2026 FIFA World Cup future fact test, if the response contains objective success criteria, any critic violation stating it failed is discarded."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = json.dumps({
        "score": 0.0,
        "issues": [
            {"description": "The response does not explicitly state that the 2026 FIFA World Cup has not happened or concluded yet, so the winner is unknown.", "severity": "Critical", "confidence": "High"},
            {"description": "The response lacks professional tone.", "severity": "Medium", "confidence": "High"}
        ],
        "suggestions": []
    })

    evaluator = CriticEvaluator(threshold=0.80)
    result = evaluator.evaluate(
        generated_text="I'm not aware of the winner of the 2026 FIFA World Cup as my knowledge cutoff is December 2023, and the 2026 FIFA World Cup has not yet occurred.",
        reference_text="The 2026 FIFA World Cup has not happened or concluded yet, so the winner is unknown.",
        user_query="Which country won the 2026 FIFA World Cup?"
    )

    # The 2026 FIFA omission issue should be discarded.
    # The 'lacks professional tone' issue (Medium) should remain.
    assert len(result.issues) == 1
    assert "lacks professional tone" in result.issues[0]
    assert result.score == 0.85
    assert result.passed is True

