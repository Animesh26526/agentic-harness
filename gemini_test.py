import time
import os
from harness.agent.gemini_agent import GeminiAgent

if __name__ == "__main__":
    os.environ["GEMINI_PACING_DELAY"] = "5.0"
    agent = GeminiAgent()
    
    print("Starting pacing test. Setting GEMINI_PACING_DELAY to 5.0 seconds.")
    for i in range(3):
        start = time.time()
        print(f"Request {i+1} starting...")
        res = agent.generate("Say hello")
        end = time.time()
        print(f"Request {i+1} finished in {end - start:.2f} seconds. Response: {res}")