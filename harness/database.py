import os
import sqlite3
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

class DatabaseManager:
    """
    Manages SQLite database interactions for the Agentic Harness benchmark runs,
    logs, and evaluation traces.
    """
    def __init__(self, db_path: str = "harness_metrics.db"):
        """
        Initializes the database manager.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to "harness_metrics.db".
        """
        # Ensure relative database paths are resolved relative to the project root directory
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(project_root, db_path)

        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "DatabaseManager":
        """Context manager entry point."""
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point with safe resource release."""
        self.close()

    def initialize(self) -> None:
        """
        Initializes the database connection and creates tables if they are missing.
        """
        if not self._conn:
            # Verify database directory exists before SQLite initialization
            db_dir = os.path.dirname(os.path.abspath(self.db_path))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            # Enable foreign key support
            self._conn.execute("PRAGMA foreign_keys = ON;")
            # Enable row factory for returning records as dictionary-like objects
            self._conn.row_factory = sqlite3.Row

        with self._conn:
            # Table 1: benchmark_runs
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    harness_enabled INTEGER,
                    avg_reliability REAL,
                    success_rate REAL,
                    total_samples INTEGER,
                    total_retries INTEGER
                );
            """)

            # Table 2: run_logs
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS run_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    query_id TEXT,
                    category TEXT,
                    query_text TEXT,
                    harness_enabled INTEGER,
                    raw_response TEXT,
                    final_response TEXT,
                    semantic_score REAL,
                    rule_score REAL,
                    critic_score REAL,
                    overall_reliability REAL,
                    retry_count INTEGER,
                    status TEXT,
                    issues TEXT,
                    FOREIGN KEY(run_id) REFERENCES benchmark_runs(run_id)
                );
            """)

            # Table 3: evaluation_traces
            try:
                self._conn.execute("ALTER TABLE evaluation_traces ADD COLUMN memory_assisted INTEGER DEFAULT 0;")
                self._conn.execute("ALTER TABLE evaluation_traces ADD COLUMN memory_strategies_json TEXT;")
                self._conn.execute("ALTER TABLE evaluation_traces ADD COLUMN negative_transfer INTEGER DEFAULT 0;")
            except sqlite3.OperationalError:
                pass
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    query_id TEXT,
                    attempt INTEGER,
                    raw_response TEXT,
                    semantic_score REAL,
                    rule_score REAL,
                    critic_score REAL,
                    overall_reliability REAL,
                    issues TEXT,
                    retry_triggered INTEGER,
                    critic_feedback TEXT,
                    memory_assisted INTEGER DEFAULT 0,
                    memory_strategies_json TEXT,
                    negative_transfer INTEGER DEFAULT 0
                );
            """)
            try:
                self._conn.execute("ALTER TABLE evaluation_traces ADD COLUMN critic_feedback TEXT;")
            except sqlite3.OperationalError:
                pass

    def create_run(
        self,
        run_id: str,
        harness_enabled: bool,
        avg_reliability: float = 0.0,
        success_rate: float = 0.0,
        total_samples: int = 0,
        total_retries: int = 0
    ) -> None:
        """
        Creates or updates a benchmark run summary.

        Args:
            run_id (str): Unique identifier for this run.
            harness_enabled (bool): Whether the self-correcting harness was enabled.
            avg_reliability (float): Aggregate reliability score for the run.
            success_rate (float): Success rate percentage (0.0 to 1.0).
            total_samples (int): Total queries evaluated in this run.
            total_retries (int): Cumulative retry requests triggered.
        """
        if not self._conn:
            self.initialize()
            
        assert self._conn is not None

        query = """
            INSERT OR REPLACE INTO benchmark_runs (
                run_id, harness_enabled, avg_reliability, success_rate, total_samples, total_retries
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        with self._conn:
            self._conn.execute(
                query,
                (
                    run_id,
                    1 if harness_enabled else 0,
                    avg_reliability,
                    success_rate,
                    total_samples,
                    total_retries
                )
            )

    def log_run_result(
        self,
        run_id: str,
        query_id: str,
        category: str,
        query_text: str,
        harness_enabled: bool,
        raw_response: str,
        final_response: str,
        semantic_score: Optional[float],
        rule_score: Optional[float],
        critic_score: Optional[float],
        overall_reliability: float,
        retry_count: int,
        status: str,
        issues: List[str]
    ) -> None:
        """
        Logs a single query result for a benchmark run.

        Args:
            run_id (str): Associated benchmark run ID.
            query_id (str): Unique query sample identifier.
            category (str): Problem category (e.g. structured_json).
            query_text (str): Input prompt.
            harness_enabled (bool): Whether the self-correcting harness was enabled.
            raw_response (str): Original first-pass response.
            final_response (str): Final returned response.
            semantic_score (Optional[float]): Final semantic similarity score.
            rule_score (Optional[float]): Final deterministic rule validation score.
            critic_score (Optional[float]): Final LLM critic evaluation score.
            overall_reliability (float): Combined overall reliability score.
            retry_count (int): Number of correction loops executed.
            status (str): Outcome status (e.g. "SUCCESS", "FAILED").
            issues (List[str]): List of detected validation issues/violations.
        """
        if not self._conn:
            self.initialize()

        assert self._conn is not None

        issues_json = json.dumps(issues)
        query = """
            INSERT INTO run_logs (
                run_id, query_id, category, query_text, harness_enabled,
                raw_response, final_response, semantic_score, rule_score,
                critic_score, overall_reliability, retry_count, status, issues
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn:
            self._conn.execute(
                query,
                (
                    run_id,
                    query_id,
                    category,
                    query_text,
                    1 if harness_enabled else 0,
                    raw_response,
                    final_response,
                    semantic_score,
                    rule_score,
                    critic_score,
                    overall_reliability,
                    retry_count,
                    status,
                    issues_json
                )
            )

    def log_trace(
        self,
        run_id: str,
        query_id: str,
        attempt: int,
        raw_response: str,
        semantic_score: Optional[float],
        rule_score: Optional[float],
        critic_score: Optional[float],
        overall_reliability: float,
        issues: List[str],
        retry_triggered: bool,
        critic_feedback: Optional[str] = None,
        memory_assisted: bool = False,
        memory_strategies_json: Optional[str] = None,
        negative_transfer: bool = False
    ) -> None:
        """
        Logs a single trace point inside a retry/correction attempt loop.

        Args:
            run_id (str): Associated benchmark run ID.
            query_id (str): Unique query sample identifier.
            attempt (int): Loop iteration index (1-based).
            raw_response (str): Response retrieved in this specific attempt.
            semantic_score (Optional[float]): Semantic score for this attempt.
            rule_score (Optional[float]): Rule score for this attempt.
            critic_score (Optional[float]): Critic score for this attempt.
            overall_reliability (float): Reliability score computed for this attempt.
            issues (List[str]): Validation issues detected in this attempt.
            retry_triggered (bool): Whether another loop retry was triggered.
            critic_feedback (str, optional): Raw critic rationale/metadata.
        """
        if not self._conn:
            self.initialize()

        assert self._conn is not None

        with self._conn:
            self._conn.execute("""
                INSERT INTO evaluation_traces (
                    run_id, query_id, attempt, raw_response,
                    semantic_score, rule_score, critic_score,
                    overall_reliability, issues, retry_triggered, critic_feedback, 
                    memory_assisted, memory_strategies_json, negative_transfer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                query_id,
                attempt,
                raw_response,
                semantic_score,
                rule_score,
                critic_score,
                overall_reliability,
                json.dumps(issues) if isinstance(issues, list) else issues,
                int(retry_triggered),
                critic_feedback,
                int(memory_assisted),
                memory_strategies_json,
                int(negative_transfer)
            ))

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a benchmark run summary from the database.

        Args:
            run_id (str): Unique identifier of the run to fetch.

        Returns:
            Optional[Dict[str, Any]]: Dictionary of run properties if found, else None.
        """
        if not self._conn:
            self.initialize()

        assert self._conn is not None

        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_run_logs(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all query logs associated with a specific benchmark run.

        Args:
            run_id (str): Run identifier to filter logs by.

        Returns:
            List[Dict[str, Any]]: List of dictionary representations of query logs.
        """
        if not self._conn:
            self.initialize()

        assert self._conn is not None

        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM run_logs WHERE run_id = ?", (run_id,))
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("issues"):
                try:
                    d["issues"] = json.loads(d["issues"])
                except Exception:
                    pass
            results.append(d)
        return results

    def get_traces(self, run_id: str, query_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all trace items tracking correction attempts for a specific query.

        Args:
            run_id (str): Benchmark run ID.
            query_id (str): Query identifier.

        Returns:
            List[Dict[str, Any]]: Ordered list of traces for each attempt.
        """
        if not self._conn:
            self.initialize()

        assert self._conn is not None

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM evaluation_traces WHERE run_id = ? AND query_id = ? ORDER BY attempt ASC",
            (run_id, query_id)
        )
        rows = cursor.fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("issues"):
                try:
                    d["issues"] = json.loads(d["issues"])
                except Exception:
                    pass
            results.append(d)
        return results

    def clear_all_data(self) -> None:
        """Wipes all rows from all tables in the SQLite database."""
        if not self._conn:
            self.initialize()
        assert self._conn is not None
        with self._conn:
            self._conn.execute("DELETE FROM evaluation_traces;")
            self._conn.execute("DELETE FROM run_logs;")
            self._conn.execute("DELETE FROM benchmark_runs;")
            try:
                self._conn.execute("DELETE FROM harness_memory;")
            except sqlite3.OperationalError:
                pass
        
        # Set isolation_level to None to guarantee autocommit mode for VACUUM
        old_isolation = self._conn.isolation_level
        self._conn.isolation_level = None
        try:
            self._conn.execute("VACUUM;")
        finally:
            self._conn.isolation_level = old_isolation

    def close(self) -> None:
        """
        Closes the database connection safely.
        """
        if self._conn:
            self._conn.close()
            self._conn = None
