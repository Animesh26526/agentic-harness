import pytest
from unittest.mock import MagicMock, patch
from harness.evaluators.rule_based import RuleBasedValidator
from harness.scoring import compute_reliability
from harness.orchestrator import Orchestrator
from harness.database import DatabaseManager

def test_character_boundary_cases():
    """Test character limits: 1 over, 2 over, 5 over, and exactly on boundary."""
    validator = RuleBasedValidator(max_length=20)
    
    # 1. Exactly on boundary (20 characters)
    text_exact = "12345678901234567890"  # 20 chars
    res_exact = validator.evaluate(text_exact)
    assert res_exact.passed is True
    assert res_exact.score == 1.0
    assert len(res_exact.issues) == 0
    
    # Verify scoring logic doesn't cap exactly on boundary
    rel_exact = compute_reliability("structured_json", rule_score=res_exact.score)
    assert rel_exact["passed"] is True
    assert rel_exact["overall_score"] == 1.0
    
    # 2. 1 character over limit (21 characters)
    text_over_1 = "123456789012345678901"  # 21 chars
    res_over_1 = validator.evaluate(text_over_1)
    assert res_over_1.passed is False
    assert res_over_1.score == 0.7
    assert any("Response length is 21" in iss and "Maximum allowed is 20" in iss and "Reduce by at least 1 characters" in iss for iss in res_over_1.issues)
    
    # Verify overall reliability score is capped below 0.80 (specifically 0.49)
    rel_over_1 = compute_reliability("structured_json", rule_score=res_over_1.score)
    assert rel_over_1["passed"] is False
    assert rel_over_1["overall_score"] == 0.49
    
    # 3. 2 characters over limit (22 characters)
    text_over_2 = "1234567890123456789012"  # 22 chars
    res_over_2 = validator.evaluate(text_over_2)
    assert res_over_2.passed is False
    assert res_over_2.score == 0.7
    assert any("Response length is 22" in iss and "Maximum allowed is 20" in iss and "Reduce by at least 2 characters" in iss for iss in res_over_2.issues)
    
    # Verify overall reliability score is capped below 0.80 (specifically 0.49)
    rel_over_2 = compute_reliability("structured_json", rule_score=res_over_2.score)
    assert rel_over_2["passed"] is False
    assert rel_over_2["overall_score"] == 0.49
    
    # 4. 5 characters over limit (25 characters)
    text_over_5 = "1234567890123456789012345"  # 25 chars
    res_over_5 = validator.evaluate(text_over_5)
    assert res_over_5.passed is False
    assert res_over_5.score == 0.7
    assert any("Response length is 25" in iss and "Maximum allowed is 20" in iss and "Reduce by at least 5 characters" in iss for iss in res_over_5.issues)
    
    # Verify overall reliability score is capped below 0.80 (specifically 0.49)
    rel_over_5 = compute_reliability("structured_json", rule_score=res_over_5.score)
    assert rel_over_5["passed"] is False
    assert rel_over_5["overall_score"] == 0.49


@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a temporary SQLite database manager."""
    db_file = tmp_path / "test_challenge.db"
    manager = DatabaseManager(db_path=str(db_file))
    manager.initialize()
    yield manager
    manager.close()


@patch('harness.orchestrator.GeminiAgent')
def test_successful_repair_behavior_for_length(mock_agent_class, temp_db):
    """Verify successful repair behavior for length violation."""
    mock_agent = mock_agent_class.return_value
    
    # Attempt 1: 21 characters (1 char over limit of 20)
    # Attempt 2: 20 characters (exactly on boundary, should pass)
    mock_agent.generate.side_effect = [
        "123456789012345678901",
        "12345678901234567890"
    ]
    
    orchestrator = Orchestrator(agent=mock_agent, db_manager=temp_db)
    
    result = orchestrator.execute(
        query="Generate a short text.",
        category="structured_json",  # structured_json is easy for rule evaluation
        evaluation_config={
            "max_length": 20
        },
        harness_enabled=True,
        run_id="run_repair_length",
        query_id="query_repair_length"
    )
    
    assert result.passed is True
    assert result.retry_count == 1
    assert result.raw_response == "123456789012345678901"
    assert result.final_response == "12345678901234567890"
    assert result.overall_score == 1.0
    
    # Verify repair prompt received warning message with correct format
    assert mock_agent.generate.call_count == 2
    repair_call_prompt = mock_agent.generate.call_args_list[1][0][0]
    assert "Response length is 21" in repair_call_prompt
    assert "Maximum allowed is 20" in repair_call_prompt
    assert "Reduce by at least 1 characters" in repair_call_prompt
