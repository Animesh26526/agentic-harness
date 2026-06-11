import pytest
from harness.evaluators.rule_based import RuleBasedValidator

def test_empty_response():
    """Confirms that empty or whitespace responses are rejected instantly."""
    validator = RuleBasedValidator()
    result = validator.evaluate("")
    
    assert result.passed is False
    assert result.score == 0.0
    assert "Response is empty." in result.issues

def test_valid_json():
    """Confirms that clean JSON strings are correctly parsed and validated."""
    validator = RuleBasedValidator()
    result = validator.evaluate(
        '{"status": "success", "data": [1, 2, 3]}',
        validate_json=True,
        required_fields=["status", "data"]
    )
    
    assert result.passed is True
    assert result.score == 1.0
    assert len(result.issues) == 0

def test_invalid_json():
    """Confirms that corrupted JSON yields a zero score and descriptive parsing errors."""
    validator = RuleBasedValidator()
    result = validator.evaluate(
        '{"status": "success", "data": [1, 2',
        validate_json=True
    )
    
    assert result.passed is False
    assert result.score == 0.0
    assert any("JSON validation failed" in issue for issue in result.issues)

def test_required_field_missing():
    """Confirms score deduction penalties for missing keys in valid JSON."""
    validator = RuleBasedValidator()
    result = validator.evaluate(
        '{"status": "success"}',
        validate_json=True,
        required_fields=["status", "data", "token"]
    )
    
    assert result.passed is False
    # Deducts 0.25 * 2 = 0.50 points
    assert result.score == 0.50
    assert any("Missing required fields: data, token" in issue for issue in result.issues)

def test_forbidden_keyword_detected():
    """Confirms deduction penalties for whole-word matches on banned words."""
    validator = RuleBasedValidator()
    result = validator.evaluate(
        "This product is quite expensive but has high durability.",
        forbidden_keywords=["expensive"]
    )
    
    assert result.passed is False
    # Deducts 0.2 points
    assert result.score == 0.80
    assert any("contains forbidden words: expensive" in issue for issue in result.issues)

def test_json_type_mismatch():
    """Confirms schema type checks and exact format reporting (expected type X but received Y)."""
    validator = RuleBasedValidator()
    result = validator.evaluate(
        '{"name": "Alice", "age": "thirty", "city": "Boston"}',
        validate_json=True,
        field_types={"name": "string", "age": "integer", "city": "string"}
    )
    
    assert result.passed is False
    assert result.score == 0.75  # Deducts 0.25
    assert "Field 'age' expected type int but received str" in result.issues

def test_length_limit_violations():
    """Confirms that responses exceeding the max characters limit get a 0.3 deduction and fail."""
    validator = RuleBasedValidator(max_length=20)
    result = validator.evaluate("This sentence is longer than 20 characters.")
    
    assert result.passed is False
    assert result.score == 0.70
    assert any("Maximum allowed is 20" in issue for issue in result.issues)

def test_markdown_wrapped_json():
    """Confirms that JSON wrapped inside Markdown formatting (```json ... ```) is cleaned and validated."""
    validator = RuleBasedValidator()
    wrapped_content = """```json
    {
      "name": "Alice",
      "age": 30
    }
    ```"""
    
    result = validator.evaluate(
        wrapped_content,
        validate_json=True,
        required_fields=["name", "age"],
        field_types={"name": "string", "age": "integer"}
    )
    
    assert result.passed is True
    assert result.score == 1.0
    assert len(result.issues) == 0

def test_json_with_surrounding_text():
    """Confirms that JSON surrounded by conversational leading/trailing text is correctly extracted and validated."""
    validator = RuleBasedValidator()
    content_with_text = """Here is the customer data:
{
  "name": "Alice",
  "age": 30
}
Let me know if you need anything else!"""

    result = validator.evaluate(
        content_with_text,
        validate_json=True,
        required_fields=["name", "age"],
        field_types={"name": "string", "age": "integer"}
    )

    assert result.passed is True
    assert result.score == 1.0
    assert len(result.issues) == 0

def test_detailed_metadata_counts():
    """Verifies that missing_fields_count, type_errors_count, and schema_errors_count are correct."""
    validator = RuleBasedValidator()
    
    # 1. Invalid JSON parse error
    result_parse = validator.evaluate("{invalid json", validate_json=True)
    assert result_parse.passed is False
    assert result_parse.metadata["schema_errors_count"] == 1
    assert result_parse.metadata["missing_fields_count"] == 0
    assert result_parse.metadata["type_errors_count"] == 0

    # 2. Missing fields only
    result_missing = validator.evaluate(
        '{"name": "Alice"}',
        validate_json=True,
        required_fields=["name", "age", "location"]
    )
    assert result_missing.passed is False
    assert result_missing.metadata["schema_errors_count"] == 2
    assert result_missing.metadata["missing_fields_count"] == 2
    assert result_missing.metadata["type_errors_count"] == 0

    # 3. Type mismatches only
    result_types = validator.evaluate(
        '{"name": 123, "age": "thirty"}',
        validate_json=True,
        field_types={"name": "string", "age": "integer"}
    )
    assert result_types.passed is False
    assert result_types.metadata["schema_errors_count"] == 2
    assert result_types.metadata["missing_fields_count"] == 0
    assert result_types.metadata["type_errors_count"] == 2

    # 4. Mixed schema errors
    result_mixed = validator.evaluate(
        '{"name": 123}',
        validate_json=True,
        required_fields=["name", "age"],
        field_types={"name": "string", "age": "integer"}
    )
    assert result_mixed.passed is False
    # age is missing (counts as missing field, but not type checked since it's missing)
    # name is type mismatch (counts as type error)
    # Total schema errors = missing (1) + type errors (1) = 2
    assert result_mixed.metadata["schema_errors_count"] == 2
    assert result_mixed.metadata["missing_fields_count"] == 1
    assert result_mixed.metadata["type_errors_count"] == 1

def test_word_count_limits():
    """Verify that min_words and max_words constraints are enforced and reported correctly."""
    validator = RuleBasedValidator()
    
    # Test min_words met
    res = validator.evaluate("One two three four five", min_words=5)
    assert res.passed is True
    assert res.score == 1.0
    
    # Test min_words failed
    res2 = validator.evaluate("One two three four", min_words=5)
    assert res2.passed is False
    assert res2.score == 0.70
    assert any("Response contains 4 words. Minimum required is 5 words." in issue for issue in res2.issues)

    # Test max_words met
    res3 = validator.evaluate("One two three four five", max_words=5)
    assert res3.passed is True
    assert res3.score == 1.0

    # Test max_words failed
    res4 = validator.evaluate("One two three four five six", max_words=5)
    assert res4.passed is False
    assert res4.score == 0.70
    assert any("Response contains 6 words. Maximum allowed is 5 words." in issue for issue in res4.issues)

def test_rule_multiple_code_blocks_json():
    """Verify that RuleBasedValidator successfully extracts and parses JSON even if it is preceded by a python block."""
    validator = RuleBasedValidator()
    response_text = """Here is some explanation code:
    ```python
    def dummy():
        return "not JSON"
    ```
    And here is the JSON:
    ```json
    {"scores": [10, 20, 30]}
    ```"""
    
    res = validator.evaluate(
        response_text,
        validate_json=True,
        required_fields=["scores"],
        field_types={"scores": "array"}
    )
    
    assert res.passed is True
    assert res.score == 1.0
