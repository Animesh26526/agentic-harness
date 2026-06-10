import os
import sqlite3
import json
from typing import Any, Dict, Optional
from harness.config import Config

class ResponseCacheManager:
    """
    Manages lightweight, SQLite-based response caching for the Agentic Harness.
    Caches execution results by prompt, model name, and harness mode.
    """
    def __init__(self, db_path: str = "harness_metrics.db"):
        """
        Initializes the cache manager.

        Args:
            db_path (str): Path to the SQLite database.
        """
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(project_root, db_path)
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Creates the cache table and indices if they do not exist."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS response_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt TEXT NOT NULL,
                        model TEXT NOT NULL,
                        harness_enabled INTEGER NOT NULL,
                        cached_result TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                # Create a unique index for quick lookups
                conn.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_response_cache 
                    ON response_cache(prompt, model, harness_enabled);
                """)
        finally:
            conn.close()

    def get(self, prompt: str, model: str, harness_enabled: bool) -> Optional[Dict[str, Any]]:
        """
        Retrieves a cached execution result if it exists.

        Args:
            prompt (str): User query prompt.
            model (str): Name of the model.
            harness_enabled (bool): Whether the harness was active.

        Returns:
            Optional[Dict[str, Any]]: The cached result dict or None.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            harness_flag = 1 if harness_enabled else 0
            cursor = conn.execute(
                """
                SELECT cached_result FROM response_cache
                WHERE prompt = ? AND model = ? AND harness_enabled = ?
                """,
                (prompt.strip(), model.strip(), harness_flag)
            )
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["cached_result"])
                except Exception:
                    return None
            return None
        finally:
            conn.close()

    def set(self, prompt: str, model: str, harness_enabled: bool, result: Dict[str, Any]) -> None:
        """
        Caches an execution result. Overwrites existing keys.

        Args:
            prompt (str): User query prompt.
            model (str): Name of the model.
            harness_enabled (bool): Whether the harness was active.
            result (Dict[str, Any]): The result dictionary to cache.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            harness_flag = 1 if harness_enabled else 0
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO response_cache (
                        prompt, model, harness_enabled, cached_result, timestamp
                    ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        prompt.strip(),
                        model.strip(),
                        harness_flag,
                        json.dumps(result)
                    )
                )
        finally:
            conn.close()

    def clear(self) -> None:
        """Clears all cached records."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM response_cache;")
        finally:
            conn.close()
