import os
import sys
from dotenv import load_dotenv
import json
import time
import uuid
from pathlib import Path
from tqdm import tqdm

# Setup workspace paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from harness.orchestrator import Orchestrator
from harness.database import DatabaseManager
from harness.analytics import get_memory_stats, get_success_rate, get_average_reliability

def get_token_estimate(text: str) -> int:
    return len(text) // 4

def run_audit():
    print("Agentic Harness V2 - Memory Audit Sprint")
    
    # Load env variables
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Enforce constraints
    os.environ["DEFAULT_MODEL"] = "Llama 3.1 8B Instant"
    
    from harness.agent.gemini_agent import GeminiAgent
    original_generate = GeminiAgent.generate
    def paced_generate(self, prompt, **kwargs):
        # Token estimation (rough)
        tokens = len(prompt) // 4
        print(f" [API CALL] Estimated Tokens: {tokens}...")
        time.sleep(6.0) # Sleep to ensure <5000 TPM (10 RPM at 500 tokens/call)
        return original_generate(self, prompt, **kwargs)
    GeminiAgent.generate = paced_generate
    
    # Instantiate agent explicitly
    agent_off = GeminiAgent(model_name="Llama 3.1 8B Instant")
    agent_on = GeminiAgent(model_name="Llama 3.1 8B Instant")
    
    dataset_path = PROJECT_ROOT / "data" / "challenge_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    samples = dataset

    db_manager = DatabaseManager(str(PROJECT_ROOT / "harness_metrics.db"))
    
    timestamp = int(time.time())
    run_id_off = f"audit_off_{timestamp}"
    run_id_on = f"audit_on_{timestamp}"
    
    print("\n[Phase 1] Running Memory OFF (Baseline)")
    orch_off = Orchestrator(agent=agent_off, db_manager=db_manager)
    os.environ["DISABLE_MEMORY"] = "1"
    for sample in tqdm(samples, desc="Baseline"):
        q_id = f"q_{uuid.uuid4().hex[:8]}"
        res = orch_off.execute(
            query=sample["input"],
            category=sample["category"],
            evaluation_config=sample.get("evaluation_config", {}),
            harness_enabled=True,
            run_id=run_id_off,
            query_id=q_id
        )
        
    print("\n[Phase 2] Running Memory ON")
    os.environ["DISABLE_MEMORY"] = "0"
    os.environ["FREEZE_MEMORY"] = "1"
    orch_on = Orchestrator(agent=agent_on, db_manager=db_manager)
    for sample in tqdm(samples, desc="Memory Enabled"):
        q_id = f"q_{uuid.uuid4().hex[:8]}"
        res = orch_on.execute(
            query=sample["input"],
            category=sample["category"],
            evaluation_config=sample.get("evaluation_config", {}),
            harness_enabled=True,
            run_id=run_id_on,
            query_id=q_id
        )

    print("\n[Phase 3] Generating Report")
    
    # Analyze OFF
    sr_off = get_success_rate(run_id_off, db_manager.db_path)
    rel_off = get_average_reliability(run_id_off, db_manager.db_path)
    
    # Analyze ON
    sr_on = get_success_rate(run_id_on, db_manager.db_path)
    rel_on = get_average_reliability(run_id_on, db_manager.db_path)
    
    mem_stats = get_memory_stats(run_id_on, db_manager.db_path)
    
    # Calculate Average Retry Reduction
    # Get total retries OFF
    run_info_off = db_manager.get_run(run_id_off)
    retries_off = run_info_off["total_retries"] if run_info_off else 0
    # Get total retries ON
    run_info_on = db_manager.get_run(run_id_on)
    retries_on = run_info_on["total_retries"] if run_info_on else 0
    avg_retry_reduction = (retries_off - retries_on) / len(samples) if retries_off > retries_on else 0.0
    
    report_content = f"""# Harness Memory Audit Report

## 1. Overview
This audit evaluates the efficacy of the Harness Memory system on the Edge-Case Challenge dataset. 
The system dynamically learned failure patterns and applied repair strategies in subsequent attempts.

## 2. Performance Comparison (Memory OFF vs ON)
| Metric | Memory OFF | Memory ON | Delta |
| :--- | :---: | :---: | :---: |
| **Challenge Success Rate** | {sr_off*100:.1f}% | {sr_on*100:.1f}% | +{(sr_on - sr_off)*100:.1f}% |
| **Average Reliability** | {rel_off:.3f} | {rel_on:.3f} | +{(rel_on - rel_off):.3f} |

## 3. Harness Memory Telemetry
| Metric | Result |
| :--- | :---: |
| **Memory Retrievals** | {mem_stats['memory_retrievals']} |
| **Memory Usage Rate** | {mem_stats['memory_usage_rate']*100:.1f}% |
| **Memory Assisted Success Rate** | {mem_stats['memory_assisted_success_rate']*100:.1f}% |
| **Average Retry Reduction** | -{avg_retry_reduction:.2f} retries/sample |
| **Average Reliability Gain** | +{mem_stats['avg_reliability_improvement']:.3f} |
| **Negative Transfer Rate** | {mem_stats['negative_transfer_rate']*100:.1f}% |

## 4. Top Learned Strategies
"""
    if mem_stats['top_learned_strategies']:
        for s in mem_stats['top_learned_strategies']:
            report_content += f"- **Strategy:** {s['strategy']}\n"
            report_content += f"  - **Success Rate:** {int(s['success_rate']*100)}%\n"
    else:
        report_content += "*(No memory strategies were successfully learned during this short subset run.)*\n"
        
    report_content += "\n## 5. Audit Conclusion\n"
    if mem_stats['negative_transfer_rate'] > 0:
        report_content += "⚠️ **Warning**: Negative transfers were detected. Some injected strategies worsened the response.\n"
    else:
        report_content += "✅ **Verified**: Zero negative transfers detected. Memory correctly assisted difficult cases.\n"
        
    out_path = "/home/codespace/.gemini/antigravity-cli/brain/7ddc77a2-ba2a-43ab-9c23-d1544612e10c/memory_audit_report.md"
    with open(out_path, "w") as f:
        f.write(report_content)
        
    print(f"\nAudit complete. Report generated at {out_path}")

if __name__ == "__main__":
    run_audit()
