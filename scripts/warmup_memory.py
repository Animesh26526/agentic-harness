import os
import sys
import json
import uuid
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Setup workspace paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from harness.orchestrator import Orchestrator
from harness.database import DatabaseManager

def run_warmup():
    print("Agentic Harness V2 - Memory Warmup Phase")
    
    # Load env variables
    load_dotenv(PROJECT_ROOT / ".env")
    
    # Enforce constraints
    os.environ["DEFAULT_MODEL"] = "Llama 3.1 8B Instant"
    os.environ["FREEZE_MEMORY"] = "0"
    os.environ["DISABLE_MEMORY"] = "0"
    
    from harness.agent.gemini_agent import GeminiAgent
    original_generate = GeminiAgent.generate
    def paced_generate(self, prompt, **kwargs):
        # Pacing to respect 5000 TPM
        time.sleep(6.0) 
        return original_generate(self, prompt, **kwargs)
    GeminiAgent.generate = paced_generate
    
    agent = GeminiAgent(model_name="Llama 3.1 8B Instant")
    
    dataset_path = PROJECT_ROOT / "data" / "challenge_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    db_manager = DatabaseManager(str(PROJECT_ROOT / "harness_metrics.db"))
    orch = Orchestrator(agent=agent, db_manager=db_manager)
    
    # We will process up to 50 samples
    samples = dataset[:50]
    
    for i, sample in enumerate(tqdm(samples, desc="Warming Up Memory")):
        q_id = f"warmup_{uuid.uuid4().hex[:8]}"
        res = orch.execute(
            query=sample["input"],
            category=sample["category"],
            evaluation_config=sample.get("evaluation_config", {}),
            harness_enabled=True,
            run_id="warmup_run",
            query_id=q_id
        )
        
        # Check condition
        cursor = db_manager._conn.cursor()
        cursor.execute("SELECT failure_type, COUNT(*) FROM memory_entries GROUP BY failure_type")
        counts = dict(cursor.fetchall())
        
        # Major failure types we care about
        targets = ["max_length_exceeded", "forbidden_keyword", "invalid_json"]
        
        # Check if all targets have at least 5
        all_met = True
        for t in targets:
            if counts.get(t, 0) < 5:
                all_met = False
                break
                
        if all_met and i >= 10:  # Ensures we run at least some samples
            print("\n[Warmup Complete] Reached target memory densities for major failure types!")
            break

if __name__ == "__main__":
    run_warmup()
