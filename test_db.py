import sqlite3
import os

def get_latest_completed_benchmark_pair(db_path):
    if not os.path.exists(db_path):
        return None, None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT run_id, harness_enabled, timestamp, total_samples FROM benchmark_runs "
            "WHERE run_id LIKE 'benchmark_%' AND total_samples = 40 "
            "ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
        
        pairs = {}
        for r in rows:
            run_id = r["run_id"]
            parts = run_id.split("_")
            if len(parts) >= 3:
                ts_key = parts[2]
                mode = parts[1]
                if ts_key not in pairs:
                    pairs[ts_key] = {}
                pairs[ts_key][mode] = run_id
                
        sorted_keys = sorted(pairs.keys(), reverse=True)
        for ts_key in sorted_keys:
            if "on" in pairs[ts_key] and "off" in pairs[ts_key]:
                return pairs[ts_key]["off"], pairs[ts_key]["on"]
        return None, None
    except Exception:
        return None, None
    finally:
        if 'conn' in locals():
            conn.close()

db_path = 'harness_metrics.db'
off, on = get_latest_completed_benchmark_pair(db_path)
print("off:", off)
print("on:", on)
