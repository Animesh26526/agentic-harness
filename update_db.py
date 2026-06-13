with open("harness/database.py", "r") as f:
    code = f.read()

# Add memory_assisted to evaluation_traces creation
old_create = """                    overall_reliability REAL,
                    issues TEXT,
                    retry_triggered INTEGER,
                    critic_feedback TEXT"""
new_create = """                    overall_reliability REAL,
                    issues TEXT,
                    retry_triggered INTEGER,
                    critic_feedback TEXT,
                    memory_assisted INTEGER DEFAULT 0"""
code = code.replace(old_create, new_create)

# Add alter table to initialize to handle existing DB
alter_block = """            # Table 3: evaluation_traces"""
new_alter = """            # Table 3: evaluation_traces
            try:
                self._conn.execute("ALTER TABLE evaluation_traces ADD COLUMN memory_assisted INTEGER DEFAULT 0;")
            except sqlite3.OperationalError:
                pass  # Column already exists
"""
code = code.replace(alter_block, new_alter)

# Modify log_trace signature
old_log = """        critic_feedback: Optional[str] = None
    ) -> None:"""
new_log = """        critic_feedback: Optional[str] = None,
        memory_assisted: bool = False
    ) -> None:"""
code = code.replace(old_log, new_log)

# Modify log_trace insert
old_insert = """                        overall_reliability, issues, retry_triggered, critic_feedback
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        critic_feedback
                    )"""
new_insert = """                        overall_reliability, issues, retry_triggered, critic_feedback, memory_assisted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        int(memory_assisted)
                    )"""
code = code.replace(old_insert, new_insert)

with open("harness/database.py", "w") as f:
    f.write(code)
