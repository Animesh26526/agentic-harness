import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from harness.config import Config
from harness.agent.gemini_agent import GeminiAgent
from harness.database import DatabaseManager
from harness.evaluation_router import get_evaluators
from harness.evaluators.semantic import SemanticEvaluator, warmup
from harness.evaluators.rule_based import RuleBasedValidator
from harness.evaluators.critic import CriticEvaluator
from harness.scoring import compute_reliability
from harness.memory import HarnessMemory

@dataclass
class ExecutionResult:
    """Represents the final outcomes and metadata of an orchestrated query execution."""
    query_id: str
    category: str
    raw_response: str
    final_response: str

    semantic_score: Optional[float]
    rule_score: Optional[float]
    critic_score: Optional[float]

    overall_score: float

    retry_count: int

    passed: bool

    issues: List[str]


class Orchestrator:
    """
    Central execution controller coordinating model query generation,
    evaluation routing, reliability scoring, and self-correcting retry loops.
    """
    def __init__(
        self,
        agent: Optional[GeminiAgent] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        """
        Initializes the Orchestrator.

        Args:
            agent (GeminiAgent, optional): Custom agent. Defaults to standard GeminiAgent.
            db_manager (DatabaseManager, optional): Custom DB manager. Defaults to standard DatabaseManager.
        """
        self.agent = agent or GeminiAgent()
        self.db_manager = db_manager or DatabaseManager()
        
        # Internal cache of evaluator instances to avoid redundant initialization
        self._semantic_evaluator: Optional[SemanticEvaluator] = None
        self._rule_validator: Optional[RuleBasedValidator] = None
        self._critic_evaluator: Optional[CriticEvaluator] = None
        self.memory = HarnessMemory(db_path=self.db_manager.db_path)

        # Preload semantic model once during orchestrator startup
        warmup()

    def _get_semantic_evaluator(self) -> SemanticEvaluator:
        if self._semantic_evaluator is None:
            self._semantic_evaluator = SemanticEvaluator()
        return self._semantic_evaluator

    def _get_rule_validator(self) -> RuleBasedValidator:
        if self._rule_validator is None:
            self._rule_validator = RuleBasedValidator()
        return self._rule_validator

    def _get_critic_evaluator(self) -> CriticEvaluator:
        if self._critic_evaluator is None:
            self._critic_evaluator = CriticEvaluator(agent=self.agent)
        return self._critic_evaluator

    def execute(
        self,
        query: str,
        category: str,
        evaluation_config: Dict[str, Any],
        harness_enabled: bool = True,
        run_id: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Coordinates prompt execution. Runs evaluations, retries incorrect agent responses,
        logs telemetry, and returns final results.

        Args:
            query (str): The user prompt.
            category (str): The task category (e.g. structured_json).
            evaluation_config (Dict[str, Any]): Configurations/parameters for the evaluators.
            harness_enabled (bool): Whether self-correcting evaluation & retry loops are enabled.
            run_id (str, optional): Association benchmark run ID.
            query_id (str, optional): Unique query sample identifier.

        Returns:
            ExecutionResult: Final outcome details.
        """
        r_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        q_id = query_id or f"query_{uuid.uuid4().hex[:8]}"

        # Ensure run_id exists in benchmark_runs to satisfy foreign key constraints
        if not self.db_manager.get_run(r_id):
            self.db_manager.create_run(run_id=r_id, harness_enabled=harness_enabled)

        # STEP 1: Generate response using GeminiAgent
        initial_response = self.agent.generate(query)
        current_response = initial_response

        # STEP 2: If harness_enabled=False: Return immediately. Still log results. No evaluators. No retries.
        if not harness_enabled:
            self.db_manager.log_run_result(
                run_id=r_id,
                query_id=q_id,
                category=category,
                query_text=query,
                harness_enabled=False,
                raw_response=initial_response,
                final_response=initial_response,
                semantic_score=None,
                rule_score=None,
                critic_score=None,
                overall_reliability=0.0,
                retry_count=0,
                status="SUCCESS",
                issues=[]
            )
            return ExecutionResult(
                query_id=q_id,
                category=category,
                raw_response=initial_response,
                final_response=initial_response,
                semantic_score=None,
                rule_score=None,
                critic_score=None,
                overall_score=0.0,
                retry_count=0,
                passed=False,
                issues=[]
            )

        # STEP 3: Determine active evaluators using evaluation_router
        active_evaluators = get_evaluators(category)

        attempt = 1
        max_retries = Config.MAX_RETRIES

        # Final loop state trackers
        final_semantic_score: Optional[float] = None
        final_rule_score: Optional[float] = None
        final_critic_score: Optional[float] = None
        final_overall_score = 0.0
        final_passed = False
        final_issues: List[str] = []
        last_suggestions = []
        last_patterns = []
        last_response = initial_response
        first_score = 0.0
        used_memory_in_prompt = False
        last_memory_strategies = []
        last_score = 0.0
        last_issues_count = 0

        while True:
            semantic_score = None
            rule_score = None
            critic_score = None
            critic_feedback = None
            
            issues: List[str] = []
            suggestions: List[str] = []

            # STEP 4: Execute only required evaluators
            if "rule" in active_evaluators:
                rule_eval = self._get_rule_validator()
                rule_result = rule_eval.evaluate(
                    generated_text=current_response,
                    validate_json=evaluation_config.get("validate_json", False),
                    required_fields=evaluation_config.get("required_fields"),
                    field_types=evaluation_config.get("field_types"),
                    forbidden_keywords=evaluation_config.get("forbidden_keywords"),
                    max_length=evaluation_config.get("max_length"),
                    min_length=evaluation_config.get("min_length"),
                    min_words=evaluation_config.get("min_words"),
                    max_words=evaluation_config.get("max_words")
                )
                rule_score = rule_result.score
                issues.extend(rule_result.issues)

            if "semantic" in active_evaluators:
                semantic_eval = self._get_semantic_evaluator()
                ref_text = evaluation_config.get("reference_text", "")
                semantic_result = semantic_eval.evaluate(
                    generated_text=current_response,
                    reference_text=ref_text
                )
                semantic_score = semantic_result.score
                issues.extend(semantic_result.issues)

            if "critic" in active_evaluators:
                critic_eval = self._get_critic_evaluator()
                ref_text = evaluation_config.get("reference_text", "")
                critic_result = critic_eval.evaluate(
                    generated_text=current_response,
                    reference_text=ref_text,
                    user_query=query,
                    max_length=evaluation_config.get("max_length"),
                    min_length=evaluation_config.get("min_length"),
                    validate_json=evaluation_config.get("validate_json"),
                    required_fields=evaluation_config.get("required_fields"),
                    forbidden_keywords=evaluation_config.get("forbidden_keywords")
                )
                critic_score = critic_result.score
                issues.extend(critic_result.issues)
                for s in critic_result.metadata.get("suggestions", []):
                    if isinstance(s, dict):
                        suggestions.append(s.get("description", str(s)))
                    else:
                        suggestions.append(str(s))
                critic_feedback = critic_result.metadata.get("raw_response", "")

            # STEP 5: Pass scores into compute_reliability()
            reliability_res = compute_reliability(
                category=category,
                semantic_score=semantic_score,
                rule_score=rule_score,
                critic_score=critic_score
            )
            overall_score = reliability_res["overall_score"]
            passed = reliability_res["passed"]

            # Save state
            final_semantic_score = semantic_score
            final_rule_score = rule_score
            final_critic_score = critic_score
            final_overall_score = overall_score
            if attempt == 1:
                first_score = overall_score
            final_passed = passed
            final_issues = issues

            retry_count = attempt - 1
            # Retry conditions: Reliability score is below threshold AND at least one meaningful issue exists
            retry_triggered = not passed and len(issues) > 0 and retry_count < max_retries

            # Detect negative transfer if memory was used
            is_negative_transfer = False
            if used_memory_in_prompt and attempt > 1:
                if overall_score < last_score or len(issues) > last_issues_count:
                    is_negative_transfer = True
                    print(f"NEGATIVE TRANSFER DETECTED: Score dropped ({last_score} -> {overall_score}) or issues increased ({last_issues_count} -> {len(issues)})")

            # Log trace of attempt to evaluation_traces
            import json
            self.db_manager.log_trace(
                run_id=r_id,
                query_id=q_id,
                attempt=attempt,
                raw_response=current_response,
                semantic_score=semantic_score,
                rule_score=rule_score,
                critic_score=critic_score,
                overall_reliability=overall_score,
                issues=issues,
                retry_triggered=retry_triggered,
                critic_feedback=critic_feedback,
                memory_assisted=used_memory_in_prompt,
                memory_strategies_json=json.dumps(last_memory_strategies) if used_memory_in_prompt else None,
                negative_transfer=is_negative_transfer
            )

            rule_issues = rule_result.issues if "rule" in active_evaluators else []
            semantic_issues = semantic_result.issues if "semantic" in active_evaluators else []
            critic_issues = critic_result.issues if "critic" in active_evaluators else []
            
            rule_metadata = getattr(rule_result, "metadata", {}) if "rule" in active_evaluators else {}
            critic_metadata = getattr(critic_result, "metadata", {}) if "critic" in active_evaluators else {}
            
            import os
            current_patterns = self.memory.detect_failure_patterns(issues, rule_metadata, critic_metadata)
            memory_strategies = []
            if os.environ.get("DISABLE_MEMORY", "0") != "1":
                memory_strategies = self.memory.search(current_patterns, limit=3)

            # STEP 6: If passed or maximum retries exhausted, exit the loop
            if passed or not retry_triggered:
                break

            # PART 2: Build repair prompt for the retry engine
            repair_prompt = (
                "Your previous response failed validation.\n\n"
                f"Original Query: {query}\n\n"
                f"Previous Response:\n{current_response}\n\n"
            )
            
            if issues:
                repair_prompt += "Optimize the response to satisfy ALL constraints simultaneously.\n\n"
                used_memory_in_prompt = False
                
                if rule_issues:
                    repair_prompt += "Priority 1: Objective constraints (character limits, JSON validity, required fields)\n"
                    for issue in rule_issues:
                        repair_prompt += f"- {issue}\n"
                    repair_prompt += "\n"
                    
                if semantic_issues:
                    repair_prompt += "Priority 2: Semantic/reference improvements\n"
                    for issue in semantic_issues:
                        repair_prompt += f"- {issue}\n"
                    repair_prompt += "\n"
                    
                if memory_strategies:
                    repair_prompt += "Priority 3: Past Successful Repair Strategies\n"
                    repair_prompt += "PREVIOUS SUCCESSFUL REPAIRS\n"
                    for idx, strat in enumerate(memory_strategies, 1):
                        repair_prompt += f"{idx}. {strat['repair_strategy']}\n"
                        repair_prompt += f"Success Rate: {int(strat['success_rate'] * 100)}%\n"
                    repair_prompt += "You SHOULD reuse these strategies if applicable.\n\n"
                    used_memory_in_prompt = True
                elif critic_issues:
                    repair_prompt += "Priority 3: Style improvements\n"
                    for issue in critic_issues:
                        repair_prompt += f"- {issue}\n"
                    repair_prompt += "\n"
                    
            if suggestions:
                repair_prompt += "Suggestions:\n"
                for suggestion in suggestions:
                    repair_prompt += f"- {suggestion}\n"
                repair_prompt += "\n"
                
            repair_prompt += "Please regenerate a corrected response."

            last_suggestions = suggestions
            last_patterns = current_patterns
            last_response = current_response
            last_score = overall_score
            last_issues_count = len(issues)
            last_memory_strategies = memory_strategies if memory_strategies else []

            # Regenerate response using the repair prompt
            current_response = self.agent.generate(repair_prompt)
            attempt += 1

        # Save final result in run_logs
        status = "SUCCESS" if final_passed else "FAILED"

        self.db_manager.log_run_result(
            run_id=r_id,
            query_id=q_id,
            category=category,
            query_text=query,
            harness_enabled=True,
            raw_response=initial_response,
            final_response=current_response,
            semantic_score=final_semantic_score,
            rule_score=final_rule_score,
            critic_score=final_critic_score,
            overall_reliability=final_overall_score,
            retry_count=attempt - 1,
            status=status,
            issues=final_issues
        )
        # Store in evaluation memory foundation
        try:
            retry_occurred = attempt > 1
            freeze_memory = os.environ.get("FREEZE_MEMORY", "0") == "1"
            if harness_enabled and not freeze_memory and retry_occurred and final_passed and final_overall_score >= 0.80 and final_overall_score > first_score:
                if last_patterns and last_suggestions:
                    # Store suggestions from the previous attempt as successful strategies
                    for strategy in last_suggestions:
                        self.memory.store_repair(last_patterns, strategy, example_before=last_response, example_after=current_response)
        except Exception as e:
            print("Memory store error:", str(e))

        return ExecutionResult(
            query_id=q_id,
            category=category,
            raw_response=initial_response,
            final_response=current_response,
            semantic_score=final_semantic_score,
            rule_score=final_rule_score,
            critic_score=final_critic_score,
            overall_score=final_overall_score,
            retry_count=attempt - 1,
            passed=final_passed,
            issues=final_issues
        )
