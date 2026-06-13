#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path

# Setup environment to ONLY use Llama 3.1 8B Instant
os.environ["DEFAULT_GEMINI_MODEL"] = "Llama 3.1 8B Instant"
os.environ["DEFAULT_MODEL"] = "Llama 3.1 8B Instant"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import requests

# Rate Limiter globals
MINUTE_DURATION = 60.0
RPM_LIMIT = 15
TPM_LIMIT = 5000

minute_start_time = time.time()
api_calls = 0
tokens = 0

original_post = requests.post

def rate_limited_post(*args, **kwargs):
    global minute_start_time, api_calls, tokens
    
    # Simple token estimation
    prompt_length = 0
    if "json" in kwargs:
        prompt_length = len(str(kwargs["json"]))
    estimated_tokens = prompt_length // 4

    while True:
        now = time.time()
        elapsed = now - minute_start_time
        
        if elapsed >= MINUTE_DURATION:
            minute_start_time = now
            api_calls = 0
            tokens = 0
            elapsed = 0
            
        if api_calls < RPM_LIMIT and (tokens + estimated_tokens) <= TPM_LIMIT:
            break
            
        wait_time = MINUTE_DURATION - elapsed
        print(f"RATE LIMIT REACHED (RPM: {api_calls}/{RPM_LIMIT}, TPM: {tokens}/{TPM_LIMIT}). Waiting {wait_time:.1f}s...")
        time.sleep(max(1.0, wait_time))

    # Commit the quota
    api_calls += 1
    tokens += estimated_tokens

    # Execute request
    response = original_post(*args, **kwargs)
    
    # Add response tokens
    if response.status_code == 200:
        try:
            res_json = response.json()
            if "choices" in res_json and len(res_json["choices"]) > 0:
                output_text = res_json["choices"][0]["message"].get("content", "")
                tokens += len(output_text) // 4
        except Exception:
            pass

    return response

requests.post = rate_limited_post

from scripts.run_benchmark import run_benchmark

def main():
    print("="*60)
    print("Agentic Harness - Portfolio Benchmark Generator")
    print("Model: Llama 3.1 8B Instant")
    print("Rate Limits: 15 RPM, 5000 TPM")
    print("="*60)
    
    db_path = str(PROJECT_ROOT / "harness_metrics.db")
    
    print("\n[1/2] RUNNING STANDARD BENCHMARKS...")
    run_benchmark(
        dataset_path="data/benchmark_dataset.json",
        db_path=db_path,
        sleep_delay=1.0  # rate_limited_post handles actual limit, but keep small baseline sleep
    )
    
    print("\n[2/2] RUNNING CHALLENGE BENCHMARKS...")
    run_benchmark(
        dataset_path="data/challenge_dataset.json",
        db_path=db_path,
        sleep_delay=1.0
    )
    
    print("\n✅ Benchmark Generation Complete!")
    print("Results have been saved to SQLite and Dashboard.")

if __name__ == "__main__":
    main()
