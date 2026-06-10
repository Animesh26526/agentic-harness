import pytest
import sqlite3
from harness.analytics import (
    get_success_rate,
    get_average_reliability,
    get_error_reduction_rate,
    get_recovery_rate,
    get_category_breakdown,
    get_retry_distribution
)
from harness.database import DatabaseManager

@pytest.fixture
def temp_analytics_db(tmp_path):
    """Fixture to create and populate a temporary test database for analytics."""
    db_file = tmp_path / "test_analytics.db"
    db = DatabaseManager(db_path=str(db_file))
    db.initialize()

    run_id_off = "run_off_analytics_test"
    run_id_on = "run_on_analytics_test"

    # Create run entries
    db.create_run(run_id=run_id_off, harness_enabled=False)
    db.create_run(run_id=run_id_on, harness_enabled=True)

    # Populate log entries for Harness OFF (run_off_analytics_test)
    # Sample 1: passed
    db.log_run_result(
        run_id=run_id_off, query_id="q1", category="structured_json",
        query_text="q1 text", harness_enabled=False, raw_response="res1", final_response="res1",
        semantic_score=None, rule_score=1.0, critic_score=None, overall_reliability=1.0,
        retry_count=0, status="SUCCESS", issues=[]
    )
    # Sample 2: failed
    db.log_run_result(
        run_id=run_id_off, query_id="q2", category="structured_json",
        query_text="q2 text", harness_enabled=False, raw_response="res2", final_response="res2",
        semantic_score=None, rule_score=0.0, critic_score=None, overall_reliability=0.0,
        retry_count=0, status="FAILED", issues=["JSON validation error"]
    )
    # Sample 3: failed
    db.log_run_result(
        run_id=run_id_off, query_id="q3", category="factual_qa",
        query_text="q3 text", harness_enabled=False, raw_response="res3", final_response="res3",
        semantic_score=0.5, rule_score=None, critic_score=0.5, overall_reliability=0.5,
        retry_count=0, status="FAILED", issues=["Semantic score too low"]
    )

    # Populate log entries for Harness ON (run_on_analytics_test)
    # Sample 1: passed directly
    db.log_run_result(
        run_id=run_id_on, query_id="q1", category="structured_json",
        query_text="q1 text", harness_enabled=True, raw_response="res1", final_response="res1",
        semantic_score=None, rule_score=1.0, critic_score=None, overall_reliability=1.0,
        retry_count=0, status="SUCCESS", issues=[]
    )
    # Sample 2: failed first, but passed after 1 retry (recovered!)
    db.log_run_result(
        run_id=run_id_on, query_id="q2", category="structured_json",
        query_text="q2 text", harness_enabled=True, raw_response="res2", final_response="res2_fixed",
        semantic_score=None, rule_score=0.9, critic_score=None, overall_reliability=0.9,
        retry_count=1, status="SUCCESS", issues=[]
    )
    # Sample 3: failed even after 3 retries
    db.log_run_result(
        run_id=run_id_on, query_id="q3", category="factual_qa",
        query_text="q3 text", harness_enabled=True, raw_response="res3", final_response="res3_fail",
        semantic_score=0.6, rule_score=None, critic_score=0.6, overall_reliability=0.6,
        retry_count=3, status="FAILED", issues=["Validation error"]
    )

    db.close()
    return str(db_file)

def test_analytics_success_rate(temp_analytics_db):
    """Verify success rate computation logic."""
    db = temp_analytics_db
    # OFF run: 1 pass out of 3 total = 1/3 ~ 0.333
    assert pytest.approx(get_success_rate("run_off_analytics_test", db_path=db)) == 1.0 / 3.0
    # ON run: 2 passes out of 3 total = 2/3 ~ 0.667
    assert pytest.approx(get_success_rate("run_on_analytics_test", db_path=db)) == 2.0 / 3.0
    # Non-existent run
    assert get_success_rate("non_existent_run", db_path=db) == 0.0

def test_analytics_average_reliability(temp_analytics_db):
    """Verify average reliability score computation."""
    db = temp_analytics_db
    # OFF: (1.0 + 0.0 + 0.5) / 3 = 0.50
    assert pytest.approx(get_average_reliability("run_off_analytics_test", db_path=db)) == 0.50
    # ON: (1.0 + 0.9 + 0.6) / 3 = 0.83333
    assert pytest.approx(get_average_reliability("run_on_analytics_test", db_path=db)) == 2.5 / 3.0
    # Non-existent run
    assert get_average_reliability("non_existent_run", db_path=db) == 0.0

def test_analytics_error_reduction_rate(temp_analytics_db):
    """Verify error reduction rate computation."""
    db = temp_analytics_db
    # OFF errors: 2; ON errors: 1
    # Reduction: (2 - 1) / 2 = 0.50
    assert pytest.approx(get_error_reduction_rate("run_off_analytics_test", "run_on_analytics_test", db_path=db)) == 0.50

    # Ensure division by zero handles gracefully
    assert get_error_reduction_rate("non_existent_1", "non_existent_2", db_path=db) == 0.0

def test_analytics_recovery_rate(temp_analytics_db):
    """Verify recovery rate logic (Failed OFF -> Succeeded ON)."""
    db = temp_analytics_db
    # Failed OFF: q2, q3
    # Passed ON: q1, q2
    # Recovered (intersection): q2
    # Recovery rate: 1 recovered / 2 failed OFF = 0.50
    assert pytest.approx(get_recovery_rate("run_off_analytics_test", "run_on_analytics_test", db_path=db)) == 0.50

    # Zero failures case handling
    assert get_recovery_rate("run_on_analytics_test", "run_off_analytics_test", db_path=db) == 0.0

def test_analytics_category_breakdown(temp_analytics_db):
    """Verify category breakdown statistics calculations."""
    db = temp_analytics_db
    # ON run breakdown
    breakdown = get_category_breakdown("run_on_analytics_test", db_path=db)
    
    assert "structured_json" in breakdown
    assert breakdown["structured_json"]["total_samples"] == 2
    assert breakdown["structured_json"]["success_rate"] == 1.0
    assert pytest.approx(breakdown["structured_json"]["avg_reliability"]) == 0.95

    assert "factual_qa" in breakdown
    assert breakdown["factual_qa"]["total_samples"] == 1
    assert breakdown["factual_qa"]["success_rate"] == 0.0
    assert pytest.approx(breakdown["factual_qa"]["avg_reliability"]) == 0.60

def test_analytics_retry_distribution(temp_analytics_db):
    """Verify retry distribution counter maps."""
    db = temp_analytics_db
    # ON run retry counts: q1=0, q2=1, q3=3
    dist = get_retry_distribution("run_on_analytics_test", db_path=db)
    
    assert dist == {0: 1, 1: 1, 3: 1}
    
    # OFF run: q1=0, q2=0, q3=0
    dist_off = get_retry_distribution("run_off_analytics_test", db_path=db)
    assert dist_off == {0: 3}
