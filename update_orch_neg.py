import json

with open("harness/orchestrator.py", "r") as f:
    code = f.read()

# 1. Add tracking variables
target_init = """        last_response = initial_response
        first_score = 0.0
        used_memory_in_prompt = False"""

new_init = """        last_response = initial_response
        first_score = 0.0
        used_memory_in_prompt = False
        last_memory_strategies = []
        last_score = 0.0
        last_issues_count = 0"""
code = code.replace(target_init, new_init)

# 2. Detect negative transfer and log
target_log = """            # Log trace of attempt to evaluation_traces
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
                memory_assisted=used_memory_in_prompt
            )"""

new_log = """            # Detect negative transfer if memory was used
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
            )"""
code = code.replace(target_log, new_log)

# 3. Update variables after logging
target_update = """            last_suggestions = suggestions
            last_patterns = current_patterns
            last_response = current_response

            # Regenerate response using the repair prompt
            current_response = self.agent.generate(repair_prompt)
            attempt += 1"""

new_update = """            last_suggestions = suggestions
            last_patterns = current_patterns
            last_response = current_response
            last_score = overall_score
            last_issues_count = len(issues)
            last_memory_strategies = memory_strategies if memory_strategies else []

            # Regenerate response using the repair prompt
            current_response = self.agent.generate(repair_prompt)
            attempt += 1"""
code = code.replace(target_update, new_update)

with open("harness/orchestrator.py", "w") as f:
    f.write(code)
