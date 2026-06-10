import json
import re
from typing import Any, List, Dict, Optional
from harness.config import Config
from harness.evaluators.base_evaluator import BaseEvaluator, EvaluationResult

class RuleBasedValidator(BaseEvaluator):
    """Deterministic structural and syntax rules validator with JSON type checking."""

    # Maps string representations to python type display names for clean errors
    TYPE_DISPLAY_MAP = {
        "string": "str",
        "integer": "int",
        "float": "float",
        "number": "number",
        "boolean": "bool",
        "array": "list",
        "object": "dict"
    }

    def __init__(self, max_length: int = None):
        """
        Initializes the validator.

        Args:
            max_length (int, optional): The maximum length constraints. Defaults to config default.
        """
        self.max_length = max_length if max_length is not None else Config.MAX_RESPONSE_LENGTH

    def _validate_type(self, val: Any, expected_type_str: str) -> bool:
        """
        Validates if the python value matches the expected type string.

        Args:
            val (Any): The value to check.
            expected_type_str (str): The expected type representation (e.g. 'integer').

        Returns:
            bool: True if type matches, False otherwise.
        """
        if expected_type_str == "string":
            return isinstance(val, str)
        elif expected_type_str == "integer":
            # Note: in Python, isinstance(True, int) is True, so we must exclude bools
            return isinstance(val, int) and not isinstance(val, bool)
        elif expected_type_str in ("float", "number"):
            return isinstance(val, (int, float)) and not isinstance(val, bool)
        elif expected_type_str == "boolean":
            return isinstance(val, bool)
        elif expected_type_str == "array":
            return isinstance(val, list)
        elif expected_type_str == "object":
            return isinstance(val, dict)
        return True

    def evaluate(
        self,
        generated_text: str,
        validate_json: bool = False,
        required_fields: Optional[List[str]] = None,
        field_types: Optional[Dict[str, str]] = None,
        forbidden_keywords: Optional[List[str]] = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Evaluates the generated text against specified rules, validating types for JSON.

        Args:
            generated_text (str): Response to check.
            validate_json (bool): If True, parses the response as JSON.
            required_fields (list, optional): Keys required in parsed JSON output.
            field_types (dict, optional): Map of keys to expected type strings.
            forbidden_keywords (list, optional): Forbidden substrings or phrases.

        Returns:
            EvaluationResult: Contains a normalized score, pass/fail status, and details of violations.
        """
        issues: List[str] = []
        score = 1.0
        missing_fields_count = 0
        type_errors_count = 0
        schema_errors_count = 0

        # 1. Empty Response Detection
        if not generated_text or not generated_text.strip():
            return EvaluationResult(
                score=0.0,
                passed=False,
                issues=["Response is empty."],
                metadata={
                    "length": 0,
                    "is_json": False,
                    "schema_errors_count": 0,
                    "missing_fields_count": 0,
                    "type_errors_count": 0
                }
            )

        # 2. Maximum Length Validation
        effective_max_length = kwargs.get("max_length") if kwargs.get("max_length") is not None else self.max_length
        if len(generated_text) > effective_max_length:
            issues.append(f"Response length ({len(generated_text)}) exceeds the limit of {effective_max_length}.")
            score -= 0.3

        # 3. JSON Validity Validation
        parsed_json: Optional[Dict[str, Any]] = None
        if validate_json:
            try:
                from harness.evaluators.critic import safe_json_parse
                parsed_json = safe_json_parse(generated_text)
            except ValueError as e:
                issues.append(f"JSON validation failed: Invalid JSON format. Error: {str(e)}")
                schema_errors_count = 1
                return EvaluationResult(
                    score=0.0,
                    passed=False,
                    issues=issues,
                    metadata={
                        "length": len(generated_text),
                        "is_json": False,
                        "schema_errors_count": schema_errors_count,
                        "missing_fields_count": missing_fields_count,
                        "type_errors_count": type_errors_count
                    }
                )

            # 4. Required Field Validation
            if required_fields and parsed_json is not None:
                missing_fields = [f for f in required_fields if f not in parsed_json]
                if missing_fields:
                    issues.append(f"JSON schema violation: Missing required fields: {', '.join(missing_fields)}")
                    score -= 0.25 * len(missing_fields)
                    missing_fields_count = len(missing_fields)

            # 5. Field Type Validation
            if field_types and parsed_json is not None:
                for field, expected_type_str in field_types.items():
                    if field in parsed_json:
                        val = parsed_json[field]
                        if not self._validate_type(val, expected_type_str):
                            expected_disp = self.TYPE_DISPLAY_MAP.get(expected_type_str, expected_type_str)
                            received_disp = type(val).__name__
                            issues.append(
                                f"Field '{field}' expected type {expected_disp} but received {received_disp}"
                            )
                            score -= 0.25
                            type_errors_count += 1

            # schema_errors_count is the sum of missing required fields and type mismatches
            schema_errors_count = missing_fields_count + type_errors_count

        # 6. Forbidden Keyword Validation
        if forbidden_keywords:
            matched_keywords = []
            for kw in forbidden_keywords:
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE)
                if pattern.search(generated_text):
                    matched_keywords.append(kw)

            if matched_keywords:
                issues.append(f"Constraint violation: Response contains forbidden words: {', '.join(matched_keywords)}")
                score -= 0.2 * len(matched_keywords)

        # Clamp score between 0.0 and 1.0
        score = max(0.0, min(1.0, score))
        passed = len(issues) == 0

        return EvaluationResult(
            score=score,
            passed=passed,
            issues=issues,
            metadata={
                "length": len(generated_text),
                "is_json": parsed_json is not None,
                "schema_errors_count": schema_errors_count,
                "missing_fields_count": missing_fields_count,
                "type_errors_count": type_errors_count
            }
        )
