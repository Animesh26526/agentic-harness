import sqlite3
db_path = 'harness_metrics.db'
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute("SELECT run_id FROM benchmark_runs WHERE harness_enabled = 1 AND total_samples = 40 ORDER BY timestamp DESC LIMIT 1")
row_on = cursor.fetchone()
if row_on:
    run_on = row_on['run_id']
    ts = run_on.split('_')[2]
    cursor.execute(f"SELECT run_id FROM benchmark_runs WHERE harness_enabled = 0 AND total_samples = 40 AND run_id LIKE '%_{ts}_%' LIMIT 1")
    row_off = cursor.fetchone()
    if row_off:
        print("ON:", run_on)
        print("OFF:", row_off['run_id'])
