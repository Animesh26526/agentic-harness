import os
import sys
import json
import time
import uuid
from typing import Any, Dict, List, Optional

# Add the workspace directory to python path if not already there
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.config import Config
from harness.database import DatabaseManager
from harness.orchestrator import Orchestrator
from harness.evaluation_router import get_evaluators
from harness.evaluators.semantic import SemanticEvaluator
from harness.evaluators.rule_based import RuleBasedValidator
from harness.evaluators.critic import CriticEvaluator
from harness.scoring import compute_reliability
from harness.reporting import export_benchmark_report


from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_benchmark(
    dataset_path: str = "data/benchmark_dataset.json",
    db_path: str = "harness_metrics.db",
    sleep_delay: float = 4.0,
    max_samples: Optional[int] = None
) -> Dict[str, Any]:
    """
    Runs the benchmark dataset across Harness OFF (Mode A) and Harness ON (Mode B).
    Computes comparative metrics, updates SQLite database telemetry, and returns summary metrics.

    Args:
        dataset_path (str): Path to the benchmark dataset JSON file.
        db_path (str): SQLite database file path.
        sleep_delay (float): Seconds to sleep between sample queries to avoid rate limits.
        max_samples (int, optional): Max number of samples to evaluate.

    Returns:
        Dict[str, Any]: Compiled summary metrics of the benchmark run.
    """
    p_dataset_path = Path(dataset_path)
    if not p_dataset_path.is_absolute():
        p_dataset_path = PROJECT_ROOT / p_dataset_path
    dataset_path = str(p_dataset_path)

    p_db_path = Path(db_path)
    if not p_db_path.is_absolute():
        p_db_path = PROJECT_ROOT / p_db_path
    db_path = str(p_db_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if max_samples is not None:
        dataset = dataset[:max_samples]

    # Initialize Managers and Evaluators
    db_manager = DatabaseManager(db_path=db_path)
    db_manager.initialize()

    # Pass the database manager to Orchestrator to ensure sharing database state/connection
    orchestrator = Orchestrator(db_manager=db_manager)

    # Initialize evaluators once for Mode A manual evaluation
    semantic_eval = SemanticEvaluator()
    rule_eval = RuleBasedValidator()
    critic_eval = CriticEvaluator(agent=orchestrator.agent)

    # Generate unique run IDs for this benchmark execution
    run_timestamp = int(time.time())
    run_id_off = f"benchmark_off_{run_timestamp}_{uuid.uuid4().hex[:4]}"
    run_id_on = f"benchmark_on_{run_timestamp}_{uuid.uuid4().hex[:4]}"

    # Register initial benchmark runs
    db_manager.create_run(run_id=run_id_off, harness_enabled=False)
    db_manager.create_run(run_id=run_id_on, harness_enabled=True)

    # Aggregated Counters
    total_samples = len(dataset)
    passed_off_count = 0
    passed_on_count = 0
    reliability_scores_off: List[float] = []
    reliability_scores_on: List[float] = []
    total_retries_on = 0
    failed_off_but_passed_on_count = 0

    print(f"Starting benchmark run on {total_samples} samples...")
    print(f"Harness OFF Run ID: {run_id_off}")
    print(f"Harness ON Run ID:  {run_id_on}")
    print("-" * 50)

    for idx, sample in enumerate(dataset):
        query_id = sample.get("query_id", f"q_{idx}")
        category = sample["category"]
        query_text = sample["input"]
        eval_config = sample.get("evaluation_config", {})

        print(f"[{idx+1}/{total_samples}] Evaluating query: {query_id} ({category})")

        # -------------------------------------------------------------
        # Mode A: Harness OFF
        # -------------------------------------------------------------
        # Execute agent but return immediately without active correction
        passed_off = False
        overall_score_off = 0.0
        try:
            result_off = orchestrator.execute(
                query=query_text,
                category=category,
                evaluation_config=eval_config,
                harness_enabled=False,
                run_id=run_id_off,
                query_id=query_id
            )

            # Evaluate the baseline response manually using the active evaluators for metrics mapping
            active_evals = get_evaluators(category)
            sem_score = None
            rul_score = None
            crt_score = None
            issues_off: List[str] = []

            # Default reference text fallback to expected_output if missing in evaluation_config
            reference_text = eval_config.get("reference_text") or sample.get("expected_output", "")

            if "rule" in active_evals:
                rule_res = rule_eval.evaluate(
                    generated_text=result_off.final_response,
                    validate_json=eval_config.get("validate_json", False),
                    required_fields=eval_config.get("required_fields"),
                    field_types=eval_config.get("field_types"),
                    forbidden_keywords=eval_config.get("forbidden_keywords")
                )
                rul_score = rule_res.score
                issues_off.extend(rule_res.issues)

            if "semantic" in active_evals:
                semantic_res = semantic_eval.evaluate(
                    generated_text=result_off.final_response,
                    reference_text=reference_text
                )
                sem_score = semantic_res.score
                issues_off.extend(semantic_res.issues)

            if "critic" in active_evals:
                critic_res = critic_eval.evaluate(
                    generated_text=result_off.final_response,
                    reference_text=reference_text,
                    user_query=query_text
                )
                crt_score = critic_res.score
                issues_off.extend(critic_res.issues)

            reliability_res_off = compute_reliability(
                category=category,
                semantic_score=sem_score,
                rule_score=rul_score,
                critic_score=crt_score
            )
            overall_score_off = reliability_res_off["overall_score"]
            passed_off = reliability_res_off["passed"]

            reliability_scores_off.append(overall_score_off)
            if passed_off:
                passed_off_count += 1

            # Update the database log record with actual baseline evaluated scores
            if db_manager._conn:
                with db_manager._conn:
                    db_manager._conn.execute(
                        """
                        UPDATE run_logs
                        SET semantic_score = ?, rule_score = ?, critic_score = ?,
                            overall_reliability = ?, status = ?, issues = ?
                        WHERE run_id = ? AND query_id = ?
                        """,
                        (
                            sem_score,
                            rul_score,
                            crt_score,
                            overall_score_off,
                            "SUCCESS" if passed_off else "FAILED",
                            json.dumps(issues_off),
                            run_id_off,
                            query_id
                        )
                    )
        except Exception as e:
            print(f"  Error during Mode A (OFF) execution: {str(e)}")
            reliability_scores_off.append(0.0)
            try:
                db_manager.log_run_result(
                    run_id=run_id_off,
                    query_id=query_id,
                    category=category,
                    query_text=query_text,
                    harness_enabled=False,
                    raw_response="",
                    final_response="",
                    semantic_score=None,
                    rule_score=None,
                    critic_score=None,
                    overall_reliability=0.0,
                    retry_count=0,
                    status="FAILED",
                    issues=[f"Execution failed: {str(e)}"]
                )
            except Exception:
                pass

        # -------------------------------------------------------------
        # Mode B: Harness ON
        # -------------------------------------------------------------
        # Execute agent with full self-correcting retry loop enabled
        try:
            result_on = orchestrator.execute(
                query=query_text,
                category=category,
                evaluation_config=eval_config,
                harness_enabled=True,
                run_id=run_id_on,
                query_id=query_id
            )

            passed_on = result_on.passed
            overall_score_on = result_on.overall_score
            retry_count_on = result_on.retry_count

            reliability_scores_on.append(overall_score_on)
            if passed_on:
                passed_on_count += 1
            total_retries_on += retry_count_on

            # Track Recovery rate criteria
            if not passed_off and passed_on:
                failed_off_but_passed_on_count += 1

            print(f"  OFF: Score={overall_score_off:.2f} | Passed={passed_off}")
            print(f"  ON:  Score={overall_score_on:.2f} | Passed={passed_on} | Retries={retry_count_on}")
        except Exception as e:
            print(f"  Error during Mode B (ON) execution: {str(e)}")
            reliability_scores_on.append(0.0)
            try:
                db_manager.log_run_result(
                    run_id=run_id_on,
                    query_id=query_id,
                    category=category,
                    query_text=query_text,
                    harness_enabled=True,
                    raw_response="",
                    final_response="",
                    semantic_score=None,
                    rule_score=None,
                    critic_score=None,
                    overall_reliability=0.0,
                    retry_count=0,
                    status="FAILED",
                    issues=[f"Execution failed: {str(e)}"]
                )
            except Exception:
                pass
            print(f"  OFF: Score={overall_score_off:.2f} | Passed={passed_off}")
            print(f"  ON:  Failed due to error: {str(e)}")

        # Stay within free tier limits
        if sleep_delay > 0 and idx < total_samples - 1:
            time.sleep(sleep_delay)

    # -------------------------------------------------------------
    # Summary Metrics Calculations
    # -------------------------------------------------------------
    errors_off = total_samples - passed_off_count
    errors_on = total_samples - passed_on_count

    success_rate_off = passed_off_count / total_samples if total_samples > 0 else 0.0
    success_rate_on = passed_on_count / total_samples if total_samples > 0 else 0.0

    avg_reliability_off = sum(reliability_scores_off) / total_samples if total_samples > 0 else 0.0
    avg_reliability_on = sum(reliability_scores_on) / total_samples if total_samples > 0 else 0.0
    avg_retry_count_on = total_retries_on / total_samples if total_samples > 0 else 0.0

    error_reduction_rate = (errors_off - errors_on) / errors_off if errors_off > 0 else 0.0
    recovery_rate = failed_off_but_passed_on_count / errors_off if errors_off > 0 else 0.0

    # Write final aggregate benchmark run statistics to DB
    db_manager.create_run(
        run_id=run_id_off,
        harness_enabled=False,
        avg_reliability=avg_reliability_off,
        success_rate=success_rate_off,
        total_samples=total_samples,
        total_retries=0
    )
    db_manager.create_run(
        run_id=run_id_on,
        harness_enabled=True,
        avg_reliability=avg_reliability_on,
        success_rate=success_rate_on,
        total_samples=total_samples,
        total_retries=total_retries_on
    )

    db_manager.close()

    # Export benchmark report
    export_benchmark_report(
        run_id_off=run_id_off,
        run_id_on=run_id_on,
        total_samples=total_samples,
        db_path=db_path
    )

    metrics = {
        "total_samples": total_samples,
        "run_id_off": run_id_off,
        "run_id_on": run_id_on,
        "success_rate_off": success_rate_off,
        "success_rate_on": success_rate_on,
        "avg_reliability_off": avg_reliability_off,
        "avg_reliability_on": avg_reliability_on,
        "avg_retry_count": avg_retry_count_on,
        "error_reduction_rate": error_reduction_rate,
        "recovery_rate": recovery_rate
    }

    # Print clean summary report
    print("\n" + "=" * 50)
    print("BENCHMARK REPORT")
    print("=" * 50)
    print(f"Total Samples evaluated: {total_samples}")
    print(f"Success Rate (OFF):      {success_rate_off * 100:.1f}%")
    print(f"Success Rate (ON):       {success_rate_on * 100:.1f}%")
    print(f"Average Reliability (OFF): {avg_reliability_off:.3f}")
    print(f"Average Reliability (ON):  {avg_reliability_on:.3f}")
    print(f"Average Retry Count (ON):  {avg_retry_count_on:.2f}")
    print(f"Error Reduction Rate:      {error_reduction_rate * 100:.1f}%")
    print(f"Recovery Rate:             {recovery_rate * 100:.1f}%")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    run_benchmark()
