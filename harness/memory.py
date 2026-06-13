import os
import sqlite3
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from harness.config import Config

@dataclass
class EvaluationMemory:
    """Represents a stored evaluation memory containing prompt, response, scores, issues, and corrections."""
    prompt: str
    response: str
    semantic_score: Optional[float]
    rule_score: Optional[float]
    critic_score: Optional[float]
    overall_score: float
    issues: List[str]
    corrections: List[str]
    timestamp: Optional[str] = None
    id: Optional[int] = None

class MemoryManager:
    """
    Interface and SQLite implementation for managing prior evaluation memories.
    Serves as the foundation for future context reuse and semantic embedding search.
    """
    def __init__(self, db_path: str = "harness_metrics.db"):
        """
        Initializes the MemoryManager.
        
        Args:
            db_path (str): Path to the SQLite database.
        """
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(project_root, db_path)
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Creates the memory table if it does not already exist."""
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS harness_memory (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        semantic_score REAL,
                        rule_score REAL,
                        critic_score REAL,
                        overall_score REAL NOT NULL,
                        issues TEXT NOT NULL,          -- JSON string representing list of issues
                        corrections TEXT NOT NULL,     -- JSON string representing list of corrections/retry feedback
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                """)
        finally:
            conn.close()

    def store_evaluation(
        self,
        prompt: str,
        response: str,
        semantic_score: Optional[float],
        rule_score: Optional[float],
        critic_score: Optional[float],
        overall_score: float,
        issues: List[str],
        corrections: List[str]
    ) -> int:
        """
        Stores an evaluation run into memory.

        Args:
            prompt (str): The initial user query/prompt.
            response (str): The final generated response.
            semantic_score (Optional[float]): The semantic evaluation score.
            rule_score (Optional[float]): The rule-based evaluation score.
            critic_score (Optional[float]): The LLM critic evaluation score.
            overall_score (float): The final overall reliability score.
            issues (List[str]): List of issues identified during validation.
            corrections (List[str]): List of correction actions or suggestions made.

        Returns:
            int: The inserted record ID.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO harness_memory (
                        prompt, response, semantic_score, rule_score, critic_score,
                        overall_score, issues, corrections
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt,
                        response,
                        semantic_score,
                        rule_score,
                        critic_score,
                        overall_score,
                        json.dumps(issues),
                        json.dumps(corrections)
                    )
                )
                return cursor.lastrowid
        finally:
            conn.close()

    def retrieve_memories(self, limit: int = 20) -> List[EvaluationMemory]:
        """
        Retrieves the most recent stored evaluations from memory.

        Args:
            limit (int): The maximum number of entries to retrieve.

        Returns:
            List[EvaluationMemory]: A list of EvaluationMemory dataclass objects.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                """
                SELECT id, prompt, response, semantic_score, rule_score, critic_score,
                       overall_score, issues, corrections, timestamp
                FROM harness_memory
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )
            memories = []
            for row in cursor.fetchall():
                try:
                    issues_list = json.loads(row["issues"])
                except Exception:
                    issues_list = []
                try:
                    corrections_list = json.loads(row["corrections"])
                except Exception:
                    corrections_list = []

                memories.append(
                    EvaluationMemory(
                        id=row["id"],
                        prompt=row["prompt"],
                        response=row["response"],
                        semantic_score=row["semantic_score"],
                        rule_score=row["rule_score"],
                        critic_score=row["critic_score"],
                        overall_score=row["overall_score"],
                        issues=issues_list,
                        corrections=corrections_list,
                        timestamp=row["timestamp"]
                    )
                )
            return memories
        finally:
            conn.close()

    def find_similar_memory(self, prompt: str) -> Optional[EvaluationMemory]:
        """
        Performs a basic exact or substring match search in SQLite memory.
        Serves as the placeholder interface for future embedding similarity search.

        Args:
            prompt (str): The prompt string to query.

        Returns:
            Optional[EvaluationMemory]: The closest matching memory if found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Simple exact or substring search
            cursor = conn.execute(
                """
                SELECT id, prompt, response, semantic_score, rule_score, critic_score,
                       overall_score, issues, corrections, timestamp
                FROM harness_memory
                WHERE prompt = ? OR ? LIKE '%' || prompt || '%'
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (prompt, prompt)
            )
            row = cursor.fetchone()
            if row:
                try:
                    issues_list = json.loads(row["issues"])
                except Exception:
                    issues_list = []
                try:
                    corrections_list = json.loads(row["corrections"])
                except Exception:
                    corrections_list = []

                return EvaluationMemory(
                    id=row["id"],
                    prompt=row["prompt"],
                    response=row["response"],
                    semantic_score=row["semantic_score"],
                    rule_score=row["rule_score"],
                    critic_score=row["critic_score"],
                    overall_score=row["overall_score"],
                    issues=issues_list,
                    corrections=corrections_list,
                    timestamp=row["timestamp"]
                )
            return None
        finally:
            conn.close()
