from harness.utils import compute_response_length
from harness.evaluators.rule_based import RuleBasedValidator

def test_compute_response_length():
    assert compute_response_length("abc") == 3
    assert compute_response_length("") == 0
