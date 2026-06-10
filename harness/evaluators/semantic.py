from typing import Any
from sentence_transformers import SentenceTransformer, util
from harness.config import Config
from harness.evaluators.base_evaluator import BaseEvaluator, EvaluationResult

class SemanticEvaluator(BaseEvaluator):
    """Evaluates semantic similarity between a generated text and a ground truth target."""

    # Class-level model variable to guarantee the transformer model is loaded once
    _model: SentenceTransformer = None

    def __init__(self, threshold: float = None, cache_folder: str = None):
        """
        Initializes the semantic evaluator.

        Args:
            threshold (float, optional): Custom similarity threshold. Defaults to config default.
            cache_folder (str, optional): Custom model cache directory path. Defaults to config default.
        """
        self.threshold = threshold if threshold is not None else Config.SEMANTIC_THRESHOLD
        self.cache_folder = cache_folder or Config.MODEL_CACHE_FOLDER

        if SemanticEvaluator._model is None:
            # Load the model with cache directory configuration for local/cloud deployment
            SemanticEvaluator._model = SentenceTransformer(
                "all-MiniLM-L6-v2",
                cache_folder=self.cache_folder
            )

    def evaluate(self, generated_text: str, reference_text: str, **kwargs: Any) -> EvaluationResult:
        """
        Computes cosine similarity using SentenceTransformers utility helper.

        Args:
            generated_text (str): The response text to evaluate.
            reference_text (str): The ground truth comparison text.

        Returns:
            EvaluationResult: Resulting score (cosine similarity) and pass/fail verdict.
        """
        # Ensure we handle empty inputs gracefully
        if not generated_text or not generated_text.strip() or not reference_text or not reference_text.strip():
            return EvaluationResult(
                score=0.0,
                passed=False,
                issues=["Missing generated_text or reference_text for semantic evaluation."]
            )

        # 1. Exact match / containment check for short facts or numeric targets (QA and Math extraction)
        clean_gen = generated_text.strip().lower()
        clean_ref = reference_text.strip().lower()
        # Treat as short fact if length <= 12 or if it is numeric
        is_numeric = clean_ref.replace('.', '', 1).isdigit()
        if clean_ref in clean_gen and (len(clean_ref) <= 12 or is_numeric):
            return EvaluationResult(
                score=1.0,
                passed=True,
                issues=[],
                metadata={
                    "raw_similarity": 1.0,
                    "normalized_similarity": 1.0,
                    "threshold": self.threshold,
                    "match_type": "exact_substring"
                }
            )

        # Generate embeddings
        gen_emb = self._model.encode(generated_text, convert_to_numpy=True)
        ref_emb = self._model.encode(reference_text, convert_to_numpy=True)

        # Calculate cosine similarity using sentence-transformers util
        similarity_tensor = util.cos_sim(gen_emb, ref_emb)
        raw_similarity = float(similarity_tensor.item())

        # Clamp raw_similarity to [-1.0, 1.0] to handle float precision issues
        raw_similarity = max(-1.0, min(1.0, raw_similarity))

        # Mathematically correct normalization: -1 -> 0.0, 0 -> 0.5, 1 -> 1.0
        normalized_score = (raw_similarity + 1.0) / 2.0
        normalized_score = max(0.0, min(1.0, normalized_score))

        passed = normalized_score >= self.threshold
        issues = []

        if not passed:
            issues.append(f"Semantic similarity score {normalized_score:.3f} falls below threshold of {self.threshold:.3f}.")

        return EvaluationResult(
            score=normalized_score,
            passed=passed,
            issues=issues,
            metadata={
                "raw_similarity": raw_similarity,
                "normalized_similarity": normalized_score,
                "threshold": self.threshold
            }
        )


def warmup() -> None:
    """
    Preloads the SemanticEvaluator (SentenceTransformer) model at startup.
    This avoids first-query latency during evaluations.
    """
    # Trigger model loading by instantiating SemanticEvaluator once
    SemanticEvaluator()
