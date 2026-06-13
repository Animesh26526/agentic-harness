import os
import sys
import unittest
from pathlib import Path

# Setup workspace paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from harness.orchestrator import Orchestrator
from harness.database import DatabaseManager

class MockAgent:
    def __init__(self):
        self.prompts = []
    
    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '{"bad": "response"}'  # Always fail to trigger retry loops

class MockEvaluatorResult:
    def __init__(self, passed, issues):
        self.passed = passed
        self.issues = issues
        self.score = 1.0 if passed else 0.0
        self.metadata = {}

class MockSemantic:
    def evaluate(self, *args, **kwargs):
        return MockEvaluatorResult(False, ["Semantic failed"])

class MockRule:
    def evaluate(self, *args, **kwargs):
        return MockEvaluatorResult(False, ["Rule failed"])

class MockCritic:
    def evaluate(self, *args, **kwargs):
        return MockEvaluatorResult(False, ["Critic failed"])

class TestMemoryEmptyState(unittest.TestCase):
    def setUp(self):
        self.db_path = str(PROJECT_ROOT / "test_harness_metrics.db")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        self.db_manager = DatabaseManager(self.db_path)
        
        # Override evaluators on orchestrator instance
        self.agent_off = MockAgent()
        self.orch_off = Orchestrator(agent=self.agent_off, db_manager=self.db_manager)
        self.orch_off._semantic_evaluator = MockSemantic()
        self.orch_off._rule_validator = MockRule()
        self.orch_off._critic_evaluator = MockCritic()
        
        self.agent_on = MockAgent()
        self.orch_on = Orchestrator(agent=self.agent_on, db_manager=self.db_manager)
        self.orch_on._semantic_evaluator = MockSemantic()
        self.orch_on._rule_validator = MockRule()
        self.orch_on._critic_evaluator = MockCritic()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_empty_memory_retry_prompt_identical(self):
        os.environ["DISABLE_MEMORY"] = "1"
        self.orch_off.execute("test", "structured_json", {}, run_id="off", query_id="1")
        
        os.environ["DISABLE_MEMORY"] = "0"
        self.orch_on.execute("test", "structured_json", {}, run_id="on", query_id="2")
        
        # Ensure we ran retries
        self.assertGreater(len(self.agent_off.prompts), 1)
        self.assertGreater(len(self.agent_on.prompts), 1)
        
        # Ensure the retry prompts match exactly
        self.assertEqual(self.agent_off.prompts[1], self.agent_on.prompts[1])
        
        # Check that 'PREVIOUS SUCCESSFUL REPAIRS' is not injected
        self.assertNotIn("PREVIOUS SUCCESSFUL REPAIRS", self.agent_on.prompts[1])

if __name__ == "__main__":
    unittest.main()
