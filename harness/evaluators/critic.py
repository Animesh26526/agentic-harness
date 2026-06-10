import json
import re
from typing import Any, Dict, List, Optional
from harness.config import Config
from harness.evaluators.base_evaluator import BaseEvaluator, EvaluationResult
from harness.agent.gemini_agent import GeminiAgent

def fix_json_unescaped_quotes(text: str) -> str:
    """Escapes unescaped double quotes inside JSON string values character-by-character."""
    chars = list(text)
    in_string = False
    escaped = False
    result = []
    i = 0
    n = len(chars)
    
    while i < n:
        c = chars[i]
        
        if c == '"' and not escaped:
            if in_string:
                # We see a double quote. Is it the closing quote or an unescaped inner quote?
                # A closing quote must be followed by structural JSON chars: ",", "}", "]", ":", or EOF/whitespace.
                # Let's peek ahead to find the next non-whitespace character.
                peek = i + 1
                while peek < n and chars[peek].isspace():
                    peek += 1
                
                next_non_space = chars[peek] if peek < n else ''
                
                # Structural markers: a closing quote of a string in a JSON list or dict
                # is followed by ',', '}', ']', or ':' (if it was a key).
                if next_non_space in (',', '}', ']', ':', ''):
                    is_closing = True
                    if next_non_space == ',':
                        # Peek further to make sure the next item makes sense in JSON structure
                        peek2 = peek + 1
                        while peek2 < n and chars[peek2].isspace():
                            peek2 += 1
                        if peek2 < n and chars[peek2] not in ('"', '{', '[', ']', '}'):
                            # The next character after ',' is not starting a new JSON value/key/structure!
                            # So this is probably an unescaped quote followed by a comma inside a string!
                            is_closing = False
                            
                    if is_closing:
                        in_string = False
                        result.append(c)
                    else:
                        result.append('\\"')
                else:
                    # Unescaped inner quote! Escape it.
                    result.append('\\"')
            else:
                in_string = True
                result.append(c)
        else:
            if c == '\\' and not escaped:
                escaped = True
            else:
                escaped = False
            result.append(c)
        i += 1
        
    return "".join(result)

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Parses JSON safely from a string response, applying cleanup and recovery rules.

    Args:
        text (str): Raw string content containing JSON.

    Returns:
        Dict[str, Any]: Parsed JSON dictionary.

    Raises:
        ValueError: If JSON cannot be parsed after all recovery attempts.
    """
    if not text or not text.strip():
        raise ValueError("Empty input string")

    cleaned = text.strip()

    # 1. Direct try
    try:
        return json.loads(fix_json_unescaped_quotes(cleaned))
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Markdown block cleanup
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        try:
            return json.loads(fix_json_unescaped_quotes(extracted))
        except (json.JSONDecodeError, ValueError):
            cleaned = extracted  # Fall back to cleanup on the extracted substring

    # 3. Find first '{' and last '}'
    start_idx = cleaned.find('{')
    end_idx = cleaned.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = cleaned[start_idx:end_idx + 1]
        try:
            return json.loads(fix_json_unescaped_quotes(candidate))
        except (json.JSONDecodeError, ValueError):
            cleaned = candidate  # Use candidate for further cleanup

    # 4. Lightweight recovery for common LLM syntax errors
    # E.g. Trailing commas before closing braces/brackets
    recovered = re.sub(r',\s*}', '}', cleaned)
    recovered = re.sub(r',\s*\]', ']', recovered)
    try:
        return json.loads(fix_json_unescaped_quotes(recovered))
    except (json.JSONDecodeError, ValueError):
        pass

    # Replace single quotes with double quotes as a last resort
    try:
        single_quote_recovered = recovered.replace("'", '"')
        return json.loads(fix_json_unescaped_quotes(single_quote_recovered))
    except (json.JSONDecodeError, ValueError):
        pass

    raise ValueError("All JSON parsing and recovery attempts failed")


class CriticEvaluator(BaseEvaluator):
    """
    LLM-as-a-Judge evaluator using Gemini to critique response adherence,
    identifying instruction following, unsupported claims, missing info, and formatting issues.
    """

    def __init__(self, threshold: Optional[float] = None, agent: Optional[GeminiAgent] = None):
        """
        Initializes the CriticEvaluator.

        Args:
            threshold (float, optional): Custom pass/fail threshold. Defaults to Config.CRITIC_THRESHOLD.
            agent (GeminiAgent, optional): Custom Gemini agent instance. If None, instantiates a default.
        """
        self.threshold = threshold if threshold is not None else Config.CRITIC_THRESHOLD
        self.agent = agent or GeminiAgent()

    def evaluate(
        self,
        generated_text: str,
        reference_text: str,
        user_query: str,
        **kwargs: Any
    ) -> EvaluationResult:
        """
        Critiques the generated response against reference text and user query using the Gemini agent.

        Args:
            generated_text (str): Response output to judge.
            reference_text (str): Ground truth reference text.
            user_query (str): The original user query.

        Returns:
            EvaluationResult: Contains a score, pass/fail status, issues, and metadata.
        """
        # Guard against empty/missing inputs
        if not generated_text or not generated_text.strip():
            return EvaluationResult(
                score=0.0,
                passed=False,
                issues=["Generated text is empty."],
                metadata={"raw_response": ""}
            )

        if reference_text and reference_text.strip():
            prompt = (
                "Critique this generated response against the user query and ground truth reference text.\n\n"
                f"Query: {user_query}\n"
                f"Reference: {reference_text}\n"
                f"Response: {generated_text}\n\n"
                "Return ONLY a JSON object with keys:\n"
                "- \"score\": float (0.0 to 1.0 representing instruction/factual adherence)\n"
                "- \"issues\": list of strings detailing errors/claims unsupported by reference or query constraints\n"
                "- \"suggestions\": list of repair suggestions to fix the issues\n\n"
                "CRITICAL GUIDELINE FOR ISSUES:\n"
                "- Do NOT hallucinate violations. Only report a forbidden word as a violation if it literally and explicitly appears in the Response.\n"
                "- Do NOT criticize words that are merely associated with or implying a forbidden word unless the query explicitly bans related concepts.\n"
                "- Be objective and literal. Avoid overly pedantic or subjective claims.\n\n"
                "IMPORTANT: Your response MUST be valid JSON. Ensure that all nested double quotes inside the strings "
                "(like quote marks or measurements such as 6.7\") are properly escaped with a backslash as \\\" so the JSON parses successfully."
            )
        else:
            prompt = (
                "Critique this generated response against the user query instructions and constraints.\n\n"
                f"Query: {user_query}\n"
                f"Response: {generated_text}\n\n"
                "Return ONLY a JSON object with keys:\n"
                "- \"score\": float (0.0 to 1.0 representing instruction/constraint adherence)\n"
                "- \"issues\": list of strings detailing failures to follow instructions/constraints (e.g. length limits, forbidden words, tone)\n"
                "- \"suggestions\": list of repair suggestions to fix the issues\n\n"
                "CRITICAL GUIDELINE FOR ISSUES:\n"
                "- Do NOT hallucinate violations. Only report a forbidden word as a violation if it literally and explicitly appears in the Response.\n"
                "- Do NOT criticize words that are merely associated with or implying a forbidden word unless the query explicitly bans related concepts.\n"
                "- Be objective and literal. Avoid overly pedantic or subjective claims.\n\n"
                "IMPORTANT: Your response MUST be valid JSON. Ensure that all nested double quotes inside the strings "
                "(like quote marks or measurements such as 6.7\") are properly escaped with a backslash as \\\" so the JSON parses successfully."
            )

        try:
            raw_response = self.agent.generate(prompt)
        except Exception as e:
            return EvaluationResult(
                score=0.0,
                passed=False,
                issues=[f"Gemini API call failed during evaluation: {str(e)}"],
                metadata={"error": str(e)}
            )

        try:
            parsed_data = safe_json_parse(raw_response)
        except Exception:
            return EvaluationResult(
                score=0.0,
                passed=False,
                issues=["Critic returned invalid JSON"],
                metadata={"raw_response": raw_response}
            )

        score = parsed_data.get("score")
        issues = parsed_data.get("issues")
        suggestions = parsed_data.get("suggestions")

        # Validation of the parsed JSON format
        validation_issues: List[str] = []
        
        if score is None:
            score = 0.0
            validation_issues.append("Critic response missing required 'score' field")
        else:
            try:
                score = float(score)
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = 0.0
                validation_issues.append("Critic response 'score' field is not a number")

        if issues is None:
            issues = []
            validation_issues.append("Critic response missing required 'issues' field")
        elif not isinstance(issues, list):
            issues = [str(issues)]
            validation_issues.append("Critic response 'issues' field is not a list")

        if suggestions is None:
            suggestions = []
            validation_issues.append("Critic response missing required 'suggestions' field")
        elif not isinstance(suggestions, list):
            suggestions = [str(suggestions)]
            validation_issues.append("Critic response 'suggestions' field is not a list")

        # Combine parsed issues and validation issues
        all_issues = [str(issue) for issue in issues] + validation_issues

        passed = score >= self.threshold and len(validation_issues) == 0

        return EvaluationResult(
            score=score,
            passed=passed,
            issues=all_issues,
            metadata={
                "suggestions": suggestions,
                "raw_response": raw_response,
                "threshold": self.threshold
            }
        )
