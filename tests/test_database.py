import os
import tempfile
import pytest
from harness.database import DatabaseManager

@pytest.fixture
def temp_db():
    """Fixture to create and clean up a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    manager = DatabaseManager(db_path=path)
    manager.initialize()
    
    yield manager
    
    manager.close()
    if os.path.exists(path):
        os.remove(path)

def test_db_initialization(temp_db):
    """Test database tables are initialized successfully."""
    conn = temp_db._conn
    assert conn is not None
    
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    assert "benchmark_runs" in tables
    assert "run_logs" in tables
    assert "evaluation_traces" in tables

def test_create_run(temp_db):
    """Test creating, updating, and retrieving a benchmark run."""
    run_id = "test_run_001"
    
    # Insert new run
    temp_db.create_run(
        run_id=run_id,
        harness_enabled=True,
        avg_reliability=0.85,
        success_rate=0.90,
        total_samples=10,
        total_retries=3
    )
    
    # Retrieve the run
    run = temp_db.get_run(run_id)
    assert run is not None
    assert run["run_id"] == run_id
    assert run["harness_enabled"] == 1
    assert run["avg_reliability"] == 0.85
    assert run["success_rate"] == 0.90
    assert run["total_samples"] == 10
    assert run["total_retries"] == 3
    
    # Update the run
    temp_db.create_run(
        run_id=run_id,
        harness_enabled=True,
        avg_reliability=0.92,
        success_rate=0.95,
        total_samples=12,
        total_retries=4
    )
    
    updated_run = temp_db.get_run(run_id)
    assert updated_run is not None
    assert updated_run["avg_reliability"] == 0.92
    assert updated_run["total_samples"] == 12

def test_log_run_result(temp_db):
    """Test logging and retrieving run results."""
    run_id = "test_run_002"
    query_id = "q_001"
    
    # Create the run first to satisfy foreign key constraints
    temp_db.create_run(run_id=run_id, harness_enabled=False)
    
    # Log result
    temp_db.log_run_result(
        run_id=run_id,
        query_id=query_id,
        category="structured_json",
        query_text="Provide a list of users.",
        harness_enabled=False,
        raw_response='{"users": []}',
        final_response='{"users": []}',
        semantic_score=1.0,
        rule_score=1.0,
        critic_score=None,
        overall_reliability=1.0,
        retry_count=0,
        status="SUCCESS",
        issues=["First issue", "Second issue"]
    )
    
    # Fetch run logs
    logs = temp_db.get_run_logs(run_id)
    assert len(logs) == 1
    log = logs[0]
    
    assert log["run_id"] == run_id
    assert log["query_id"] == query_id
    assert log["category"] == "structured_json"
    assert log["query_text"] == "Provide a list of users."
    assert log["harness_enabled"] == 0
    assert log["raw_response"] == '{"users": []}'
    assert log["final_response"] == '{"users": []}'
    assert log["semantic_score"] == 1.0
    assert log["rule_score"] == 1.0
    assert log["critic_score"] is None
    assert log["overall_reliability"] == 1.0
    assert log["retry_count"] == 0
    assert log["status"] == "SUCCESS"
    assert log["issues"] == ["First issue", "Second issue"]

def test_log_trace(temp_db):
    """Test logging and retrieving evaluation traces."""
    run_id = "test_run_003"
    query_id = "q_002"
    
    # Log some attempts
    temp_db.log_trace(
        run_id=run_id,
        query_id=query_id,
        attempt=1,
        raw_response="Attempt 1 output",
        semantic_score=0.4,
        rule_score=0.5,
        critic_score=0.3,
        overall_reliability=0.4,
        issues=["Low semantic similarity", "Missing age field"],
        retry_triggered=True
    )
    
    temp_db.log_trace(
        run_id=run_id,
        query_id=query_id,
        attempt=2,
        raw_response="Attempt 2 output",
        semantic_score=0.9,
        rule_score=1.0,
        critic_score=0.8,
        overall_reliability=0.9,
        issues=[],
        retry_triggered=False
    )
    
    # Retrieve traces
    traces = temp_db.get_traces(run_id, query_id)
    assert len(traces) == 2
    
    assert traces[0]["attempt"] == 1
    assert traces[0]["raw_response"] == "Attempt 1 output"
    assert traces[0]["overall_reliability"] == 0.4
    assert traces[0]["issues"] == ["Low semantic similarity", "Missing age field"]
    assert traces[0]["retry_triggered"] == 1
    
    assert traces[1]["attempt"] == 2
    assert traces[1]["raw_response"] == "Attempt 2 output"
    assert traces[1]["overall_reliability"] == 0.9
    assert traces[1]["issues"] == []
    assert traces[1]["retry_triggered"] == 0

def test_context_manager_safety():
    """Test that context manager closes the database connection automatically."""
    temp_file = "temp_context_test.db"
    
    if os.path.exists(temp_file):
        os.remove(temp_file)
        
    try:
        with DatabaseManager(temp_file) as db:
            db.create_run("run_context_001", True)
            assert db._conn is not None
            run = db.get_run("run_context_001")
            assert run is not None
            
        assert db._conn is None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
