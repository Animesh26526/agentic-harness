import sqlite3
import json
from typing import Dict, Any, List

def _get_connection(db_path: str) -> sqlite3.Connection:
    """Helper to open connection and set row factory."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_success_rate(run_id: str, db_path: str = "harness_metrics.db") -> float:
    """
    Computes success rate (fraction of queries with 'SUCCESS' status) for a run.

    Args:
        run_id (str): The run identifier.
        db_path (str): Path to the SQLite database.

    Returns:
        float: Success rate between 0.0 and 1.0.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as passed "
            "FROM run_logs WHERE run_id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        if not row or row["total"] == 0:
            return 0.0
        return float(row["passed"]) / float(row["total"])
    finally:
        conn.close()

def get_average_reliability(run_id: str, db_path: str = "harness_metrics.db") -> float:
    """
    Computes average reliability score for a run.

    Args:
        run_id (str): The run identifier.
        db_path (str): Path to the SQLite database.

    Returns:
        float: Average reliability score.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT AVG(overall_reliability) as avg_reliability FROM run_logs WHERE run_id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        if not row or row["avg_reliability"] is None:
            return 0.0
        return float(row["avg_reliability"])
    finally:
        conn.close()

def get_error_reduction_rate(run_id_off: str, run_id_on: str, db_path: str = "harness_metrics.db") -> float:
    """
    Computes error reduction rate from a Harness OFF baseline to a Harness ON run.
    Formula: (errors_off - errors_on) / errors_off

    Args:
        run_id_off (str): Baseline run ID (Harness OFF).
        run_id_on (str): Active run ID (Harness ON).
        db_path (str): Path to the SQLite database.

    Returns:
        float: Error reduction rate.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        # Get errors in OFF run
        cursor.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as passed "
            "FROM run_logs WHERE run_id = ?",
            (run_id_off,)
        )
        row_off = cursor.fetchone()
        errors_off = (row_off["total"] - row_off["passed"]) if (row_off and row_off["total"]) else 0

        # Get errors in ON run
        cursor.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as passed "
            "FROM run_logs WHERE run_id = ?",
            (run_id_on,)
        )
        row_on = cursor.fetchone()
        errors_on = (row_on["total"] - row_on["passed"]) if (row_on and row_on["total"]) else 0

        if errors_off == 0:
            return 0.0

        return float(errors_off - errors_on) / float(errors_off)
    finally:
        conn.close()

def get_recovery_rate(run_id_off: str, run_id_on: str, db_path: str = "harness_metrics.db") -> float:
    """
    Computes recovery rate (fraction of queries that failed in Harness OFF but passed in Harness ON).

    Args:
        run_id_off (str): Baseline run ID (Harness OFF).
        run_id_on (str): Active run ID (Harness ON).
        db_path (str): Path to the SQLite database.

    Returns:
        float: Recovery rate.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        # Find failed query_ids in baseline run
        cursor.execute("SELECT query_id FROM run_logs WHERE run_id = ? AND status != 'SUCCESS'", (run_id_off,))
        failed_off = {row["query_id"] for row in cursor.fetchall()}

        if not failed_off:
            return 0.0

        # Find successful query_ids in active run
        cursor.execute("SELECT query_id FROM run_logs WHERE run_id = ? AND status = 'SUCCESS'", (run_id_on,))
        passed_on = {row["query_id"] for row in cursor.fetchall()}

        # Intersection: failed off but passed on
        recovered = failed_off.intersection(passed_on)
        return float(len(recovered)) / float(len(failed_off))
    finally:
        conn.close()

def get_category_breakdown(run_id: str, db_path: str = "harness_metrics.db") -> Dict[str, Dict[str, Any]]:
    """
    Computes success rate, average reliability, and total samples per task category.

    Args:
        run_id (str): The run identifier.
        db_path (str): Path to the SQLite database.

    Returns:
        Dict[str, Dict[str, Any]]: Category breakdown statistics.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT category, COUNT(*) as total, "
            "SUM(CASE WHEN status = 'SUCCESS' THEN 1 ELSE 0 END) as passed, "
            "AVG(overall_reliability) as avg_rel "
            "FROM run_logs WHERE run_id = ? "
            "GROUP BY category",
            (run_id,)
        )
        rows = cursor.fetchall()
        breakdown = {}
        for row in rows:
            cat = row["category"]
            total = row["total"]
            passed = row["passed"]
            avg_rel = row["avg_rel"]
            breakdown[cat] = {
                "success_rate": float(passed) / float(total) if total > 0 else 0.0,
                "avg_reliability": float(avg_rel) if avg_rel is not None else 0.0,
                "total_samples": int(total)
            }
        return breakdown
    finally:
        conn.close()

def get_retry_distribution(run_id: str, db_path: str = "harness_metrics.db") -> Dict[int, int]:
    """
    Computes retry distribution (retry_count maps to number of queries).

    Args:
        run_id (str): The run identifier.
        db_path (str): Path to the SQLite database.

    Returns:
        Dict[int, int]: Retry count distribution mapping.
    """
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT retry_count, COUNT(*) as qty "
            "FROM run_logs WHERE run_id = ? "
            "GROUP BY retry_count ORDER BY retry_count ASC",
            (run_id,)
        )
        rows = cursor.fetchall()
        # Default distribution dict
        distribution = {}
        for row in rows:
            distribution[int(row["retry_count"])] = int(row["qty"])
        return distribution
    finally:
        conn.close()
