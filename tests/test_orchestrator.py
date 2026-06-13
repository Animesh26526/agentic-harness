import pytest
from unittest.mock import MagicMock, patch
from harness.orchestrator import Orchestrator, ExecutionResult
from harness.database import DatabaseManager
from harness.config import Config

@pytest.fixture
def temp_db(tmp_path):
    """Fixture providing a temporary SQLite database manager."""
    db_file = tmp_path / "test_harness.db"
    manager = DatabaseManager(db_path=str(db_file))
    manager.initialize()
    yield manager
    manager.close()

@patch('harness.orchestrator.GeminiAgent')
def test_harness_off_path(mock_agent_class, temp_db):
    """Verify Harness OFF path: generates once, returns immediately, logs, and runs no evaluators."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = "Test response"

    orchestrator = Orchestrator(agent=mock_agent, db_manager=temp_db)
    
    result = orchestrator.execute(
        query="Explain relativity",
        category="factual_qa",
        evaluation_config={"reference_text": "Albert Einstein developed relativity."},
        harness_enabled=False,
        run_id="run_off_123",
        query_id="query_off_123"
    )

    assert isinstance(result, ExecutionResult)
    assert result.raw_response == "Test response"
    assert result.final_response == "Test response"
    assert result.passed is False
    assert result.overall_score == 0.0
    assert result.retry_count == 0
    assert result.semantic_score is None

    # Verify log entry in DB
    logs = temp_db.get_run_logs("run_off_123")
    assert len(logs) == 1
    assert logs[0]["harness_enabled"] == 0
    assert logs[0]["query_id"] == "query_off_123"
    
    # Verify no traces were recorded
    traces = temp_db.get_traces("run_off_123", "query_off_123")
    assert len(traces) == 0

    mock_agent.generate.assert_called_once_with("Explain relativity")

@patch('harness.orchestrator.GeminiAgent')
def test_harness_on_pass_path(mock_agent_class, temp_db):
    """Verify Harness ON path where the first response passes validation."""
    mock_agent = mock_agent_class.return_value
    mock_agent.generate.return_value = '{"name": "Alice", "age": 30, "city": "Boston"}'

    orchestrator = Orchestrator(agent=mock_agent, db_manager=temp_db)
    
    result = orchestrator.execute(
        query="Create JSON Alice 30 Boston",
        category="structured_json",
        evaluation_config={
            "validate_json": True,
            "required_fields": ["name", "age", "city"]
        },
        harness_enabled=True,
        run_id="run_on_pass",
        query_id="query_on_pass"
    )

    assert result.passed is True
    assert result.overall_score == 1.0
    assert result.retry_count == 0
    assert result.raw_response == '{"name": "Alice", "age": 30, "city": "Boston"}'

    # Verify database log and trace
    logs = temp_db.get_run_logs("run_on_pass")
    assert len(logs) == 1
    assert logs[0]["status"] == "SUCCESS"
    assert logs[0]["retry_count"] == 0

    traces = temp_db.get_traces("run_on_pass", "query_on_pass")
    assert len(traces) == 1
    assert traces[0]["attempt"] == 1
    assert traces[0]["retry_triggered"] == 0

@patch('harness.orchestrator.GeminiAgent')
def test_harness_on_retry_and_pass_path(mock_agent_class, temp_db):
    """Verify Harness ON path where response fails initially, then passes after one retry."""
    mock_agent = mock_agent_class.return_value
    # First response is invalid json; second is valid json
    mock_agent.generate.side_effect = [
        "This is not JSON",
        '{"name": "Alice", "age": 30, "city": "Boston"}'
    ]

    orchestrator = Orchestrator(agent=mock_agent, db_manager=temp_db)
    
    result = orchestrator.execute(
        query="Create JSON",
        category="structured_json",
        evaluation_config={
            "validate_json": True,
            "required_fields": ["name", "age", "city"]
        },
        harness_enabled=True,
        run_id="run_on_retry",
        query_id="query_on_retry"
    )

    assert result.passed is True
    assert result.retry_count == 1
    assert result.raw_response == "This is not JSON"
    assert result.final_response == '{"name": "Alice", "age": 30, "city": "Boston"}'

    # Verify logs
    logs = temp_db.get_run_logs("run_on_retry")
    assert len(logs) == 1
    assert logs[0]["status"] == "SUCCESS"
    assert logs[0]["retry_count"] == 1

    # Verify traces show first failure and second success
    traces = temp_db.get_traces("run_on_retry", "query_on_retry")
    assert len(traces) == 2
    assert traces[0]["attempt"] == 1
    assert traces[0]["retry_triggered"] == 1
    assert traces[1]["attempt"] == 2
    assert traces[1]["retry_triggered"] == 0

    # Ensure agent was called with repair prompt
    assert mock_agent.generate.call_count == 2
    second_call_prompt = mock_agent.generate.call_args_list[1][0][0]
    assert "Your previous response failed validation." in second_call_prompt
    assert "Priority 1: Objective constraints" in second_call_prompt
    assert "This is not JSON" in second_call_prompt

@patch('harness.orchestrator.GeminiAgent')
def test_harness_on_fail_after_max_retries(mock_agent_class, temp_db):
    """Verify Harness ON path halts retry attempts and records FAILED status after Config.MAX_RETRIES."""
    mock_agent = mock_agent_class.return_value
    # Keep generating invalid JSON
    mock_agent.generate.return_value = "Always invalid JSON"

    orchestrator = Orchestrator(agent=mock_agent, db_manager=temp_db)
    
    result = orchestrator.execute(
        query="Create JSON",
        category="structured_json",
        evaluation_config={"validate_json": True},
        harness_enabled=True,
        run_id="run_max_fail",
        query_id="query_max_fail"
    )

    assert result.passed is False
    # Initial attempt (0) + 3 retries = 4 attempts total
    assert result.retry_count == Config.MAX_RETRIES

    # Verify database log and trace records
    logs = temp_db.get_run_logs("run_max_fail")
    assert len(logs) == 1
    assert logs[0]["status"] == "FAILED"
    assert logs[0]["retry_count"] == Config.MAX_RETRIES

    traces = temp_db.get_traces("run_max_fail", "query_max_fail")
    assert len(traces) == Config.MAX_RETRIES + 1
    for i in range(Config.MAX_RETRIES):
        assert traces[i]["retry_triggered"] == 1
    assert traces[Config.MAX_RETRIES]["retry_triggered"] == 0
