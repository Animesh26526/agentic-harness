from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class EvaluationResult:
    """Standardized representation of an evaluation run result."""
    score: float
    passed: bool
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseEvaluator(ABC):
    """Abstract base class representing an Evaluator module."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> EvaluationResult:
        """
        Executes evaluation logic on the provided arguments.

        Returns:
            EvaluationResult: The structured outcome of this evaluator execution.
        """
        pass
