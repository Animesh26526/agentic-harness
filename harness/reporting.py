import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from harness.analytics import (
    get_success_rate,
    get_average_reliability,
    get_error_reduction_rate,
    get_recovery_rate
)

# Resolve project root relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def export_benchmark_report(
    run_id_off: str,
    run_id_on: str,
    total_samples: int,
    db_path: str = "harness_metrics.db",
    output_path: str = "reports/latest_report.json"
) -> Dict[str, Any]:
    """
    Compiles comparative metrics for Harness OFF and Harness ON runs,
    exports a JSON summary report to the output path, and returns the report dictionary.

    Args:
        run_id_off (str): Run ID for the Harness OFF run.
        run_id_on (str): Run ID for the Harness ON run.
        total_samples (int): Number of evaluation samples in the benchmark.
        db_path (str): Path to the SQLite database.
        output_path (str): Target output file path.

    Returns:
        Dict[str, Any]: The generated report dictionary.
    """
    # Ensure relative paths are resolved relative to the project root
    p_db_path = Path(db_path)
    if not p_db_path.is_absolute():
        p_db_path = PROJECT_ROOT / p_db_path
    db_path = str(p_db_path)

    p_output_path = Path(output_path)
    if not p_output_path.is_absolute():
        p_output_path = PROJECT_ROOT / p_output_path
    output_path = str(p_output_path)

    # Create the reports output directory if it does not exist
    p_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute comparative analytics metrics
    success_rate_off = get_success_rate(run_id_off, db_path=db_path)
    success_rate_on = get_success_rate(run_id_on, db_path=db_path)
    
    average_reliability_off = get_average_reliability(run_id_off, db_path=db_path)
    average_reliability_on = get_average_reliability(run_id_on, db_path=db_path)
    
    error_reduction_rate = get_error_reduction_rate(run_id_off, run_id_on, db_path=db_path)
    recovery_rate = get_recovery_rate(run_id_off, run_id_on, db_path=db_path)

    # Compile report structure
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total_samples": total_samples,
        "success_rate_off": success_rate_off,
        "success_rate_on": success_rate_on,
        "average_reliability_off": average_reliability_off,
        "average_reliability_on": average_reliability_on,
        "error_reduction_rate": error_reduction_rate,
        "recovery_rate": recovery_rate
    }

    # Write JSON report file
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
