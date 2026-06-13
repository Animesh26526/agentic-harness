import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any

class HarnessMemory:
    """
    Agentic Harness V2 - Harness Memory System.
    Stores failure patterns to successful repair strategies.
    Does NOT store prompts or responses.
    """
    def __init__(self, db_path: str = "harness_metrics.db"):
        if not os.path.isabs(db_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            db_path = os.path.join(project_root, db_path)
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        failure_type TEXT,
                        subtype TEXT,
                        metadata_json TEXT,
                        repair_strategy TEXT,
                        example_before TEXT,
                        example_after TEXT,
                        success_count INTEGER DEFAULT 0,
                        failure_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        created_at TEXT
                    );
                """)
        finally:
            conn.close()

    def detect_failure_patterns(self, issues: List[str], rule_metadata: Dict, critic_metadata: Dict) -> List[Dict]:
        """
        Parses issues to detect explicit failure patterns.
        Supported types: max_length_exceeded, min_length_not_met, forbidden_keyword, 
        required_field_missing, invalid_json, hallucination, fictional_entity, contradiction.
        """
        patterns = []
        issues_str = " ".join(issues).lower()
        
        # Rule validation checks
        if "maximum length exceeded" in issues_str or "exceeds maximum length" in issues_str:
            diff = rule_metadata.get("length_difference", 0)
            patterns.append({
                "failure_type": "max_length_exceeded",
                "metadata": {"difference": diff}
            })
            
        if "minimum length not met" in issues_str or "too short" in issues_str:
            patterns.append({
                "failure_type": "min_length_not_met",
                "metadata": {}
            })
            
        if "forbidden keyword" in issues_str or "contains forbidden" in issues_str:
            patterns.append({
                "failure_type": "forbidden_keyword",
                "metadata": {}
            })
            
        if "missing required field" in issues_str or "required field" in issues_str:
            patterns.append({
                "failure_type": "required_field_missing",
                "metadata": {}
            })
            
        if "invalid json" in issues_str or "json parsing failed" in issues_str:
            patterns.append({
                "failure_type": "invalid_json",
                "metadata": {}
            })
            
        # Semantic/Critic checks
        if "hallucination" in issues_str or "not supported by reference" in issues_str:
            patterns.append({
                "failure_type": "hallucination",
                "metadata": {}
            })
            
        if "fictional entity" in issues_str:
            patterns.append({
                "failure_type": "fictional_entity",
                "metadata": {}
            })
            
        if "contradiction" in issues_str or "contradicts" in issues_str:
            patterns.append({
                "failure_type": "contradiction",
                "metadata": {}
            })

        return patterns

    def search(self, failure_patterns: List[Dict], limit: int = 3) -> List[Dict]:
        """Retrieves successful repair strategies for the given failure patterns."""
        if not failure_patterns:
            return []
            
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        types = [p["failure_type"] for p in failure_patterns]
        placeholders = ",".join("?" for _ in types)
        
        cursor.execute(f'''
            SELECT repair_strategy, success_rate, success_count 
            FROM memory_entries
            WHERE failure_type IN ({placeholders}) AND success_count > 0
            ORDER BY success_rate DESC, success_count DESC
            LIMIT ?
        ''', (*types, limit))
        
        results = [{"repair_strategy": row["repair_strategy"], "success_rate": row["success_rate"]} for row in cursor.fetchall()]
        conn.close()
        return results

    def store_repair(self, failure_patterns: List[Dict], repair_strategy: str, example_before: str = "", example_after: str = ""):
        """Stores or updates a repair strategy upon a successful retry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for pattern in failure_patterns:
            f_type = pattern["failure_type"]
            meta_json = json.dumps(pattern.get("metadata", {}))
            
            cursor.execute('''
                SELECT id, success_count, failure_count FROM memory_entries 
                WHERE failure_type = ? AND repair_strategy = ?
            ''', (f_type, repair_strategy))
            row = cursor.fetchone()
            
            if row:
                new_success = row[1] + 1
                failure_count = row[2]
                new_rate = float(new_success) / (new_success + failure_count)
                cursor.execute('''
                    UPDATE memory_entries 
                    SET success_count = ?, success_rate = ?
                    WHERE id = ?
                ''', (new_success, new_rate, row[0]))
            else:
                cursor.execute('''
                    INSERT INTO memory_entries (failure_type, metadata_json, repair_strategy, example_before, example_after, success_count, failure_count, success_rate, created_at)
                    VALUES (?, ?, ?, ?, ?, 1, 0, 1.0, ?)
                ''', (f_type, meta_json, repair_strategy, example_before, example_after, datetime.utcnow().isoformat()))
                
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Fetch memory stats for Dashboard/Playground UI."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memory_entries")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT failure_type, SUM(success_count) as c FROM memory_entries GROUP BY failure_type ORDER BY c DESC LIMIT 5")
        failures = [{"type": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        cursor.execute("SELECT repair_strategy, success_rate FROM memory_entries WHERE success_count > 0 ORDER BY success_rate DESC, success_count DESC LIMIT 5")
        top_strategies = [{"strategy": row[0], "success_rate": row[1]} for row in cursor.fetchall()]
        
        conn.close()
        return {
            "total_entries": total,
            "most_common_failures": failures,
            "top_strategies": top_strategies
        }
