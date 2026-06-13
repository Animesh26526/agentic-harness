import json
import pytest
from unittest.mock import patch, MagicMock
from harness.evaluators.base_evaluator import EvaluationResult
from harness.orchestrator import ExecutionResult
from scripts.run_benchmark import run_benchmark

@pytest.fixture
def mock_dataset_file(tmp_path):
    """Fixture providing a temporary benchmark dataset JSON file."""
    dataset = [
        {
            "query_id": "test_01",
            "category": "structured_json",
            "input": "Query 1",
            "expected_output": '{"valid": true}',
            "evaluation_config": {"validate_json": True}
        },
        {
            "query_id": "test_02",
            "category": "structured_json",
            "input": "Query 2",
            "expected_output": '{"valid": true}',
            "evaluation_config": {"validate_json": True}
        },
        {
            "query_id": "test_03",
            "category": "structured_json",
            "input": "Query 3",
            "expected_output": '{"valid": true}',
            "evaluation_config": {"validate_json": True}
        }
    ]
    file_path = tmp_path / "test_dataset.json"
    with open(file_path, "w") as f:
        json.dump(dataset, f)
    return str(file_path)

@pytest.fixture
def temp_db_file(tmp_path):
    """Fixture providing a temporary database file path."""
    return str(tmp_path / "test_metrics.db")

@patch('harness.orchestrator.GeminiAgent')
@patch('scripts.run_benchmark.RuleBasedValidator.evaluate')
@patch('scripts.run_benchmark.Orchestrator.execute')
def test_benchmark_metrics_computation(mock_execute, mock_rule_evaluate, mock_gemini_agent, mock_dataset_file, temp_db_file):
    """
    Verify benchmark execution metrics, specifically success rate, average reliability,
    error reduction rate, and recovery rate.
    
    Test scenario details:
    - 3 samples
    - Mode A (Harness OFF):
      - Sample 1: Passes (score=1.0)
      - Sample 2: Fails  (score=0.0)
      - Sample 3: Fails  (score=0.0)
      - Result: passed_off = 1, errors_off = 2, success_rate_off = 1/3 = ~0.33
    - Mode B (Harness ON):
      - Sample 1: Passes (score=1.0, retries=0)
      - Sample 2: Passes (score=1.0, retries=1) -> recovered!
      - Sample 3: Fails  (score=0.0, retries=3)
      - Result: passed_on = 2, errors_on = 1, success_rate_on = 2/3 = ~0.67
    
    Expected metrics:
    - total_samples = 3
    - error_reduction_rate = (errors_off - errors_on) / errors_off = (2 - 1) / 2 = 0.50
    - recovery_rate = recovered / errors_off = 1 / 2 = 0.50
    - avg_retry_count = (0 + 1 + 3) / 3 = 1.33
    """
    # Configure mock responses for Orchestrator execute (Mode A: harness_enabled=False, Mode B: harness_enabled=True)
    # Mode A execution return results (scores are evaluated manually from response, but execute returns mock)
    res_off_1 = ExecutionResult("test_01", "structured_json", "resp1", "resp1", None, None, None, 0.0, 0, False, [])
    res_off_2 = ExecutionResult("test_02", "structured_json", "resp2", "resp2", None, None, None, 0.0, 0, False, [])
    res_off_3 = ExecutionResult("test_03", "structured_json", "resp3", "resp3", None, None, None, 0.0, 0, False, [])

    # Mode B execution return results
    res_on_1 = ExecutionResult("test_01", "structured_json", "resp1", "resp1", None, 1.0, None, 1.0, 0, True, [])
    res_on_2 = ExecutionResult("test_02", "structured_json", "resp2", "resp2_fixed", None, 1.0, None, 1.0, 1, True, [])
    res_on_3 = ExecutionResult("test_03", "structured_json", "resp3", "resp3_failed", None, 0.0, None, 0.0, 3, False, ["fail"])

    # execute side effects:
    # idx 0: sample 1 OFF, then sample 1 ON
    # idx 1: sample 2 OFF, then sample 2 ON
    # idx 2: sample 3 OFF, then sample 3 ON
    mock_execute.side_effect = [
        res_off_1, res_on_1,
        res_off_2, res_on_2,
        res_off_3, res_on_3
    ]

    # Configure RuleBasedValidator.evaluate mock outputs (for Mode A manual validation mapping)
    eval_pass = EvaluationResult(score=1.0, passed=True, issues=[], metadata={})
    eval_fail = EvaluationResult(score=0.0, passed=False, issues=["missing field"], metadata={})
    
    mock_rule_evaluate.side_effect = [
        eval_pass,  # Sample 1 Mode A check
        eval_fail,  # Sample 2 Mode A check
        eval_fail   # Sample 3 Mode A check
    ]

    # Execute run_benchmark (set sleep_delay to 0 in tests to speed up execution)
    metrics = run_benchmark(
        dataset_path=mock_dataset_file,
        db_path=temp_db_file,
        sleep_delay=0.0
    )

    # Assertions
    assert metrics["total_samples"] == 3
    assert pytest.approx(metrics["success_rate_off"]) == 1 / 3
    assert pytest.approx(metrics["success_rate_on"]) == 2 / 3
    assert pytest.approx(metrics["avg_retry_count"]) == 4 / 3
    assert pytest.approx(metrics["error_reduction_rate"]) == 0.50
    assert pytest.approx(metrics["recovery_rate"]) == 0.50
    assert metrics["run_id_off"] != ""
    assert metrics["run_id_on"] != ""

def test_benchmark_runner_invalid_file():
    """Verify that calling run_benchmark on a non-existent dataset raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_benchmark(dataset_path="non_existent_file.json")
