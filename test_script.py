import sqlite3
import os

db_path = "harness_metrics.db"
conn = sqlite3.connect(db_path)
try:
    conn.execute("ALTER TABLE evaluation_traces ADD COLUMN memory_strategies_json TEXT;")
except sqlite3.OperationalError:
    pass

try:
    conn.execute("ALTER TABLE evaluation_traces ADD COLUMN negative_transfer INTEGER DEFAULT 0;")
except sqlite3.OperationalError:
    pass

conn.commit()
conn.close()
print("Success")
