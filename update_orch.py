import re

with open('harness/orchestrator.py', 'r') as f:
    content = f.read()

content = content.replace(
    "from harness.scoring import compute_reliability",
    "from harness.scoring import compute_reliability\nfrom harness.memory import HarnessMemory"
)

content = content.replace(
    "self._critic_evaluator: Optional[CriticEvaluator] = None",
    "self._critic_evaluator: Optional[CriticEvaluator] = None\n        self.memory = HarnessMemory(db_path=self.db_manager.db_path)"
)

# Initialize tracking variables before loop
content = content.replace(
    "final_issues: List[str] = []",
    "final_issues: List[str] = []\n        last_suggestions = []\n        last_patterns = []\n        last_response = initial_response\n        first_score = 0.0"
)

# Capture first score
content = content.replace(
    "final_overall_score = overall_score",
    "final_overall_score = overall_score\n            if attempt == 1:\n                first_score = overall_score"
)

# Detect patterns and retrieve memories right after computing reliability
old_retry_block = """            rule_issues = rule_result.issues if "rule" in active_evaluators else []
            semantic_issues = semantic_result.issues if "semantic" in active_evaluators else []
            critic_issues = critic_result.issues if "critic" in active_evaluators else []

            # STEP 6: If passed or maximum retries exhausted, exit the loop"""
new_retry_block = """            rule_issues = rule_result.issues if "rule" in active_evaluators else []
            semantic_issues = semantic_result.issues if "semantic" in active_evaluators else []
            critic_issues = critic_result.issues if "critic" in active_evaluators else []
            
            rule_metadata = getattr(rule_result, "metadata", {}) if "rule" in active_evaluators else {}
            critic_metadata = getattr(critic_result, "metadata", {}) if "critic" in active_evaluators else {}
            
            current_patterns = self.memory.detect_failure_patterns(issues, rule_metadata, critic_metadata)
            memory_strategies = self.memory.search(current_patterns, limit=3)

            # STEP 6: If passed or maximum retries exhausted, exit the loop"""
content = content.replace(old_retry_block, new_retry_block)

old_priority3 = """                if critic_issues:
                    repair_prompt += "Priority 3: Style improvements\\n"
                    for issue in critic_issues:
                        repair_prompt += f"- {issue}\\n"
                    repair_prompt += "\\n"
                    
            if suggestions:"""
new_priority3 = """                if memory_strategies:
                    repair_prompt += "Priority 3: Past Successful Repair Strategies\\n"
                    repair_prompt += "PREVIOUS SUCCESSFUL REPAIRS\\n"
                    for idx, strat in enumerate(memory_strategies, 1):
                        repair_prompt += f"{idx}. {strat['repair_strategy']}\\n"
                        repair_prompt += f"Success Rate: {int(strat['success_rate'] * 100)}%\\n"
                    repair_prompt += "You SHOULD reuse these strategies if applicable.\\n\\n"
                elif critic_issues:
                    repair_prompt += "Priority 3: Style improvements\\n"
                    for issue in critic_issues:
                        repair_prompt += f"- {issue}\\n"
                    repair_prompt += "\\n"
                    
            if suggestions:"""
content = content.replace(old_priority3, new_priority3)

# Update the tracker for last_suggestions and last_patterns right before generating
old_regen = """            # Regenerate response using the repair prompt
            current_response = self.agent.generate(repair_prompt)
            attempt += 1"""
new_regen = """            last_suggestions = suggestions
            last_patterns = current_patterns
            last_response = current_response

            # Regenerate response using the repair prompt
            current_response = self.agent.generate(repair_prompt)
            attempt += 1"""
content = content.replace(old_regen, new_regen)

# Store memory at the end
old_store = """        # Store in evaluation memory foundation
        try:
            from harness.memory import MemoryManager
            mem_manager = MemoryManager(db_path=self.db_manager.db_path)
            corrections = []
            traces = self.db_manager.get_traces(r_id, q_id)
            for t in traces:
                if t.get("issues"):
                    issues_list = []
                    import json
                    try:
                        issues_str = t["issues"]
                        if isinstance(issues_str, str):
                            issues_list = json.loads(issues_str)
                        else:
                            issues_list = issues_str
                    except Exception:
                        pass
                    for issue in issues_list:
                        if issue not in corrections:
                            corrections.append(f"Correction for: {issue}")
            mem_manager.store_evaluation(
                prompt=query,
                response=current_response,
                semantic_score=final_semantic_score,
                rule_score=final_rule_score,
                critic_score=final_critic_score,
                overall_score=final_overall_score,
                issues=final_issues,
                corrections=corrections
            )
        except Exception:
            pass"""
new_store = """        # Store in evaluation memory foundation
        try:
            retry_occurred = attempt > 1
            if harness_enabled and retry_occurred and final_passed and final_overall_score >= 0.80 and final_overall_score > first_score:
                if last_patterns and last_suggestions:
                    # Store suggestions from the previous attempt as successful strategies
                    for strategy in last_suggestions:
                        self.memory.store_repair(last_patterns, strategy, example_before=last_response, example_after=current_response)
        except Exception as e:
            print("Memory store error:", str(e))"""
content = content.replace(old_store, new_store)

with open('harness/orchestrator.py', 'w') as f:
    f.write(content)
