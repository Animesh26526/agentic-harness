import os
import json
import pytest
from datetime import datetime

from harness.reporting import export_benchmark_report
from harness.database import DatabaseManager

@pytest.fixture
def temp_report_db(tmp_path):
    """Fixture to create and populate a database for reporting tests."""
    db_file = tmp_path / "test_report.db"
    db = DatabaseManager(db_path=str(db_file))
    db.initialize()

    run_id_off = "run_off_report_test"
    run_id_on = "run_on_report_test"

    db.create_run(run_id=run_id_off, harness_enabled=False)
    db.create_run(run_id=run_id_on, harness_enabled=True)

    # OFF: 1 pass, 1 fail
    db.log_run_result(
        run_id=run_id_off, query_id="q1", category="structured_json",
        query_text="q1 text", harness_enabled=False, raw_response="res1", final_response="res1",
        semantic_score=None, rule_score=1.0, critic_score=None, overall_reliability=1.0,
        retry_count=0, status="SUCCESS", issues=[]
    )
    db.log_run_result(
        run_id=run_id_off, query_id="q2", category="structured_json",
        query_text="q2 text", harness_enabled=False, raw_response="res2", final_response="res2",
        semantic_score=None, rule_score=0.0, critic_score=None, overall_reliability=0.0,
        retry_count=0, status="FAILED", issues=["error"]
    )

    # ON: 2 passes (q2 recovered!)
    db.log_run_result(
        run_id=run_id_on, query_id="q1", category="structured_json",
        query_text="q1 text", harness_enabled=True, raw_response="res1", final_response="res1",
        semantic_score=None, rule_score=1.0, critic_score=None, overall_reliability=1.0,
        retry_count=0, status="SUCCESS", issues=[]
    )
    db.log_run_result(
        run_id=run_id_on, query_id="q2", category="structured_json",
        query_text="q2 text", harness_enabled=True, raw_response="res2", final_response="res2_fixed",
        semantic_score=None, rule_score=1.0, critic_score=None, overall_reliability=1.0,
        retry_count=1, status="SUCCESS", issues=[]
    )

    db.close()
    return str(db_file)

def test_export_benchmark_report_generation(temp_report_db, tmp_path):
    """Verify that export_benchmark_report creates the JSON report correctly."""
    report_file = tmp_path / "reports" / "latest_report.json"
    
    # Run exporter
    report = export_benchmark_report(
        run_id_off="run_off_report_test",
        run_id_on="run_on_report_test",
        total_samples=2,
        db_path=temp_report_db,
        output_path=str(report_file)
    )

    # Verify return dictionary properties
    assert isinstance(report, dict)
    assert report["total_samples"] == 2
    assert report["success_rate_off"] == 0.50
    assert report["success_rate_on"] == 1.0
    assert report["average_reliability_off"] == 0.50
    assert report["average_reliability_on"] == 1.0
    assert report["error_reduction_rate"] == 1.0  # 1 OFF error -> 0 ON errors
    assert report["recovery_rate"] == 1.0  # 1 OFF failure recovered

    # Validate timestamp parsing
    ts_str = report["timestamp"]
    assert ts_str.endswith("Z")
    # Should not raise ValueError
    datetime.fromisoformat(ts_str[:-1])

    # Verify that the JSON file was written and matches
    assert os.path.exists(str(report_file))
    with open(report_file, "r") as f:
        file_content = json.load(f)
    
    assert file_content == report
