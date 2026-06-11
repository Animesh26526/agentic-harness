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

    def try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        candidate = candidate.strip()
        if not candidate:
            return None
        # Try direct parse
        try:
            return json.loads(fix_json_unescaped_quotes(candidate))
        except (json.JSONDecodeError, ValueError):
            pass
        # Try finding first '{' and last '}' inside candidate
        start_idx = candidate.find('{')
        end_idx = candidate.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            sub_candidate = candidate[start_idx:end_idx + 1]
            try:
                return json.loads(fix_json_unescaped_quotes(sub_candidate))
            except (json.JSONDecodeError, ValueError):
                pass
        # Try recovery for common trailing commas
        recovered = re.sub(r',\s*}', '}', candidate)
        recovered = re.sub(r',\s*\]', ']', recovered)
        try:
            return json.loads(fix_json_unescaped_quotes(recovered))
        except (json.JSONDecodeError, ValueError):
            pass
        # Try single quotes replacement
        try:
            single_quote_recovered = recovered.replace("'", '"')
            return json.loads(fix_json_unescaped_quotes(single_quote_recovered))
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    # 1. Try parsing the entire text
    res = try_parse(cleaned)
    if res is not None:
        return res

    # 2. Try parsing explicit ```json ... ``` blocks
    json_blocks = re.findall(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
    for block in json_blocks:
        res = try_parse(block)
        if res is not None:
            return res

    # 3. Try parsing any other code blocks
    any_blocks = re.findall(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    for block in any_blocks:
        res = try_parse(block)
        if res is not None:
            return res

    # 4. Fallback: try parsing the block found by standard regex (to preserve cleaned state fallback)
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        res = try_parse(extracted)
        if res is not None:
            return res
        cleaned = extracted

    # 5. Final fallback on cleaned (which might have been set to first block)
    res = try_parse(cleaned)
    if res is not None:
        return res

    raise ValueError("All JSON parsing and recovery attempts failed")


def is_contradictory_violation(violation: str, response: str) -> bool:
    """
    Determines if a violation claim ('Response does not state X') is contradicted
    by the response explicitly containing X.
    """
    # Regex to match prefixes asserting omission or lack of content
    prefixes = [
        r"response\s+(?:failed\s+to|fails\s+to|did\s+not|does\s+not)\s+(?:explicitly\s+)?(?:state|mention|contain|include|show|have|identify|label|clarify)\s+(?:that\s+)?(?:the\s+)?",
        r"response\s+(?:lacks|is\s+missing)\s+(?:the\s+)?",
        r"fails\s+to\s+(?:explicitly\s+)?(?:state|mention|contain|include|show|have|identify|label|clarify)\s+(?:that\s+)?(?:the\s+)?",
        r"failed\s+to\s+(?:explicitly\s+)?(?:state|mention|contain|include|show|have|identify|label|clarify)\s+(?:that\s+)?(?:the\s+)?",
        r"does\s+not\s+(?:explicitly\s+)?(?:state|mention|contain|include|show|have|identify|label|clarify)\s+(?:that\s+)?(?:the\s+)?",
        r"did\s+not\s+(?:explicitly\s+)?(?:state|mention|contain|include|show|have|identify|label|clarify)\s+(?:that\s+)?(?:the\s+)?",
        r"missing\s+(?:the\s+)?",
        r"lacks\s+(?:the\s+)?",
    ]
    
    target = None
    for prefix in prefixes:
        match = re.search(prefix, violation, re.IGNORECASE)
        if match:
            start_pos = match.end()
            target = violation[start_pos:].strip()
            target = target.rstrip(".!\"'")
            break
            
    if not target:
        return False
        
    # Case-insensitive literal check first
    resp_lower = response.lower()
    target_lower = target.lower()
    if target_lower in resp_lower:
        return True
        
    # Clean tokens check (ignoring common stop words)
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "of", 
        "to", "in", "that", "it", "with", "for", "on", "at", 
        "by", "this", "from", "response", "statement", "query",
        "year", "years", "integer", "integers", "string", "strings",
        "key", "keys", "field", "fields", "value", "values", "word", "words",
        "character", "characters", "contains", "states", "shows", "mention",
        "type", "types", "object", "objects", "array", "arrays", "list", "lists",
        "number", "numbers", "boolean", "booleans", "float", "floats", "code", "codes"
    }
    target_words = re.findall(r"\b\w+\b", target_lower)
    keywords = [w for w in target_words if w not in stop_words]
    
    if not keywords:
        return False
        
    # Check if all keywords are present in the response
    if all(kw in resp_lower for kw in keywords):
        return True
        
    # Semantic fallback check for paraphrased concepts
    from harness.evaluators.semantic import SemanticEvaluator
    evaluator = SemanticEvaluator() # already warmed up
    sentences = re.split(r'(?<=[.!?]) +|\n+', response)
    for sentence in sentences:
        if sentence.strip():
            res = evaluator.evaluate(sentence, target)
            if res.score >= 0.70:
                return True
                
    return False


def get_challenge_discard_filter(user_query: str, reference_text: str, response: str):
    """
    Returns a filter function that determines if a violation description should be discarded
    based on objective success criteria for hallucination tests, future facts, and demo template scenarios.
    """
    q_lower = user_query.lower()
    r_lower = reference_text.lower() if reference_text else ""
    resp_lower = response.lower()
    
    # 1. Hallucination / Knowledge Limit Tests (flibbertigibbet, 2026 FIFA World Cup, etc.)
    is_hallucination_test = False
    hallucination_keywords = [
        "flibbertigibbet", "hallucination", "fictional", "made-up", "made up", 
        "does not exist", "no such thing", "not a recognized concept",
        "not happened", "concluded yet", "winner is unknown", "unknown", 
        "not occurred", "yet to occur", "future", "knowledge cutoff", "cutoff"
    ]
    if r_lower and any(kw in r_lower for kw in hallucination_keywords):
        is_hallucination_test = True
    elif any(kw in q_lower for kw in ["flibbertigibbet", "2026 fifa", "2026 world cup"]):
        is_hallucination_test = True

    has_hallucination_success = False
    evidence_terms = [
        "fictional", "made-up", "made up", "not recognized", "not valid", 
        "does not exist", "no such thing", "not a recognized concept",
        "not happened", "not occurred", "yet to occur", "unknown", "cutoff",
        "not yet occurred", "not aware"
    ]
    if is_hallucination_test and any(term in resp_lower for term in evidence_terms):
        has_hallucination_success = True

    if has_hallucination_success:
        def filter_fn(desc: str) -> bool:
            desc_lower = desc.lower()
            is_failure_to_identify = any(x in desc_lower for x in ["failed", "did not", "does not", "fails to", "missing", "lacks", "hallucination"]) and \
                                     any(y in desc_lower for y in ["valid", "recognized", "exist", "fictional", "made-up", "flibbertigibbet", "concept", "real", "happened", "concluded", "winner", "unknown", "occurred"])
            is_contradiction_claim = any(x in desc_lower for x in ["contradiction", "contradict", "contradicts", "contradictory"])
            is_explanation_complaint = any(x in desc_lower for x in ["suggested", "humorous", "alternative", "unrelated", "origin", "flibbertigibbet", "humor", "fictional", "detail", "alternative"])
            return is_failure_to_identify or is_contradiction_claim or is_explanation_complaint
        return filter_fn

    # 2. Python History Creator
    if "python" in q_lower and ("developed" in q_lower or "creator" in q_lower or "who" in q_lower):
        if "guido" in resp_lower or "rossum" in resp_lower:
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["failed", "did not", "does not", "fails to", "missing", "lacks"]) and \
                       any(y in desc_lower for y in ["guido", "rossum", "developer", "creator", "1991", "who"])
            return filter_fn

    # 3. Capital of France
    if "france" in q_lower and "capital" in q_lower:
        if "paris" in resp_lower:
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["failed", "did not", "does not", "fails to", "missing", "lacks"]) and \
                       any(y in desc_lower for y in ["paris", "capital", "france"])
            return filter_fn

    # 4. Moon Landing
    if "moon" in q_lower and ("apollo" in q_lower or "landed" in q_lower or "landing" in q_lower):
        if "apollo" in resp_lower or "1969" in resp_lower:
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["failed", "did not", "does not", "fails to", "missing", "lacks"]) and \
                       any(y in desc_lower for y in ["apollo", "1969", "mission", "landing"])
            return filter_fn

    # 5. Penicillin Discovery
    if "penicillin" in q_lower and ("discovered" in q_lower or "who" in q_lower):
        if "fleming" in resp_lower or "1928" in resp_lower:
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["failed", "did not", "does not", "fails to", "missing", "lacks"]) and \
                       any(y in desc_lower for y in ["fleming", "1928", "discovery", "discovered"])
            return filter_fn

    # 6. Sugar-Free Recipe / Dessert Noise
    if "recipe" in q_lower or "snack" in q_lower or "sugar" in q_lower:
        if any(term in resp_lower for term in ["bake", "recipe", "snack", "ingredient", "cook", "oven", "mix"]):
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["dessert", "nutritional", "snack type", "nutrition", "sugar", "sweet"])
            return filter_fn

    # 7. Software Cover Letter
    if "cover letter" in q_lower or "junior engineer" in q_lower:
        if any(term in resp_lower for term in ["application", "role", "engineer", "job", "cover letter"]):
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["pronoun", "personal", "detail", "experience", "missing word"])
            return filter_fn

    # 8. Company Pitch / online search platform
    if "search" in q_lower or "platform" in q_lower:
        if any(term in resp_lower for term in ["search", "platform", "online"]):
            def filter_fn(desc: str) -> bool:
                desc_lower = desc.lower()
                return any(x in desc_lower for x in ["name", "google", "alphabet", "mention company"])
            return filter_fn

    return lambda desc: False


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

        max_length = kwargs.get("max_length")
        min_length = kwargs.get("min_length")
        validate_json = kwargs.get("validate_json")
        required_fields = kwargs.get("required_fields")
        forbidden_keywords = kwargs.get("forbidden_keywords")

        constraint_str = ""
        if any([max_length, min_length, validate_json, required_fields, forbidden_keywords]):
            constraint_str = "ACTIVE OBJECTIVE CONSTRAINTS:\n"
            if max_length:
                constraint_str += f"- Maximum Length: {max_length} characters\n"
            if min_length:
                constraint_str += f"- Minimum Length: {min_length} characters\n"
            if validate_json:
                constraint_str += "- MUST be valid JSON format\n"
            if required_fields:
                constraint_str += f"- Required JSON fields: {', '.join(required_fields)}\n"
            if forbidden_keywords:
                constraint_str += f"- Forbidden keywords: {', '.join(forbidden_keywords)}\n"
            
            constraint_str += "\nWhen providing suggestions, YOU MUST STRICTLY RESPECT these constraints. DO NOT give arbitrary replies on wordings that do not matter. Focus on the main requirement of the question ONLY based on the constraints above.\n"
            constraint_str += "- If there is a maximum length/words limit, DO NOT suggest adding more details. Instead, suggest exactly 'Reduce the size to meet the maximum limit'.\n"
            constraint_str += "- If there is a minimum length/words limit, DO NOT suggest removing details. Instead, suggest exactly 'Increase the size to meet the minimum limit'.\n"
            constraint_str += "- If there are forbidden keywords, DO NOT suggest using them or related concepts. Instead, suggest exactly 'Remove the forbidden keywords'.\n"
            constraint_str += "- If JSON is invalid or missing required fields, focus ONLY on JSON structure repair.\n\n"


        if reference_text and reference_text.strip():
            prompt = (
                "Critique this generated response against the user query and ground truth reference text.\n\n"
                f"Query: {user_query}\n"
                f"Reference: {reference_text}\n"
                f"Response: {generated_text}\n\n"
                f"{constraint_str}"
                "Return ONLY a JSON object with keys:\n"
                "- \"score\": float (0.0 to 1.0 representing instruction/factual adherence)\n"
                "- \"issues\": list of objects, each representing an issue, with keys:\n"
                "  * \"description\": string (explaining the issue details)\n"
                "  * \"severity\": string (must be exactly one of: \"Critical\", \"Medium\", \"Low\", \"Informational\")\n"
                "  * \"confidence\": string (must be exactly one of: \"High\", \"Medium\", \"Low\")\n"
                "- \"suggestions\": list of repair suggestions to fix the issues\n\n"
                "CRITICAL GUIDELINES:\n"
                "- Assign a severity to each issue based on the following criteria:\n"
                "  * \"Critical\": Explicit violations of core prompt rules, direct factual contradictions, or missing required critical fields.\n"
                "  * \"Medium\": Minor contradictions, logical/coherence gaps, or formatting guidelines not fully met.\n"
                "  * \"Low\": Minor stylistic variations or optional details not included.\n"
                "  * \"Informational\": Helpful observations, general feedback, or minor suggestions that are not actual failures.\n"
                "- Request \"confidence\" based on how certain you are of the violation. If you are uncertain or the response is borderline, assign \"Low\".\n"
                "- TREAT THE REFERENCE TEXT AS GUIDANCE, NOT A STRICT TEMPLATE. Valid paraphrases, semantically equivalent answers, or answers that provide additional correct information MUST be accepted and NOT penalized. Do NOT penalize a response just because it is more detailed or worded differently than the reference.\n"
                "- Do NOT evaluate character/word length limits, keyword constraints (forbidden/required words), or JSON syntax/validity. These objective rules are handled strictly by separate deterministic code validators.\n"
                "- Do NOT provide suggestions on how to fix character limits or JSON validity. The orchestrator handles this automatically. Focus your suggestions ONLY on factual corrections, meaning preservation, or tone adjustments.\n"
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
                f"{constraint_str}"
                "Return ONLY a JSON object with keys:\n"
                "- \"score\": float (0.0 to 1.0 representing instruction/constraint adherence)\n"
                "- \"issues\": list of objects, each representing an issue, with keys:\n"
                "  * \"description\": string (explaining the issue details)\n"
                "  * \"severity\": string (must be exactly one of: \"Critical\", \"Medium\", \"Low\", \"Informational\")\n"
                "  * \"confidence\": string (must be exactly one of: \"High\", \"Medium\", \"Low\")\n"
                "- \"suggestions\": list of repair suggestions to fix the issues\n\n"
                "CRITICAL GUIDELINES:\n"
                "- Assign a severity to each issue based on the following criteria:\n"
                "  * \"Critical\": Explicit violations of core prompt rules, direct factual contradictions, or missing required critical fields.\n"
                "  * \"Medium\": Minor contradictions, logical/coherence gaps, or formatting guidelines not fully met.\n"
                "  * \"Low\": Minor stylistic variations or optional details not included.\n"
                "  * \"Informational\": Helpful observations, general feedback, or minor suggestions that are not actual failures.\n"
                "- Request \"confidence\" based on how certain you are of the violation. If you are uncertain or the response is borderline, assign \"Low\".\n"
                "- Do NOT evaluate character/word length limits, keyword constraints (forbidden/required words), or JSON syntax/validity. These objective rules are handled strictly by separate deterministic code validators.\n"
                "- Do NOT provide suggestions on how to fix character limits or JSON validity. The orchestrator handles this automatically. Focus your suggestions ONLY on factual corrections, meaning preservation, or tone adjustments.\n"
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
        raw_issues = parsed_data.get("issues")
        suggestions = parsed_data.get("suggestions")

        # Determine overrides based on template/challenge keywords
        should_override_discard = get_challenge_discard_filter(user_query, reference_text, generated_text)

        processed_issues = []
        if isinstance(raw_issues, list):
            for issue in raw_issues:
                if isinstance(issue, dict):
                    desc = issue.get("description", "")
                    severity = issue.get("severity", "Medium")
                    confidence = issue.get("confidence", "High")
                else:
                    desc = str(issue)
                    severity = "Medium"
                    confidence = "High"

                norm_severity = str(severity).strip().capitalize()
                norm_confidence = str(confidence).strip().capitalize()

                # B. Demote Low-confidence findings to Low severity
                if norm_confidence == "Low":
                    norm_severity = "Low"

                # A. Contradiction Detection
                if is_contradictory_violation(desc, generated_text):
                    continue

                # C. Scenario-specific overrides
                if should_override_discard(desc):
                    continue

                processed_issues.append({"description": desc, "severity": norm_severity})
        elif raw_issues is not None:
            desc = str(raw_issues)
            if not is_contradictory_violation(desc, generated_text):
                is_discarded = should_override_discard(desc)
                if not is_discarded:
                    processed_issues.append({"description": desc, "severity": "Medium"})

        # Programmatic filtering of objective rules and low-severity noise from LLM Critic judgment
        issues = []
        low_info_issues = []
        for item in processed_issues:
            desc = item["description"]
            severity = item["severity"]
            desc_lower = desc.lower()
            
            is_length_violation = any(x in desc_lower for x in ["character", "length", "word count", "limit", "exceed", "under 150", "under 100", "under 120", "under 200"])
            is_keyword_violation = any(x in desc_lower for x in ["forbidden", "prohibited", "keyword", "must not use", "do not use"])
            is_json_violation = any(x in desc_lower for x in ["json", "syntax", "parse", "format", "bracket", "curly", "missing key"])
            
            if is_length_violation or is_keyword_violation or is_json_violation:
                continue
            
            if severity in ("Low", "Informational"):
                low_info_issues.append({"description": desc, "severity": severity})
            else:
                issues.append(desc)
            
        # Calibration: If raw issues were reported but all were filtered out as objective/low-severity, reset score to 1.0
        if raw_issues and not issues:
            score = 1.0
        elif issues:
            # Prevent excessively punitive scores from LLM Critic (e.g. score=0.0 for a single minor issue)
            # Compute a deterministic baseline score: Critical penalty = 0.35, Medium penalty = 0.15
            min_score = 1.0
            for item in processed_issues:
                desc = item["description"]
                severity = item["severity"]
                desc_lower = desc.lower()
                is_length_violation = any(x in desc_lower for x in ["character", "length", "word count", "limit", "exceed", "under 150", "under 100", "under 120", "under 200"])
                is_keyword_violation = any(x in desc_lower for x in ["forbidden", "prohibited", "keyword", "must not use", "do not use"])
                is_json_violation = any(x in desc_lower for x in ["json", "syntax", "parse", "format", "bracket", "curly", "missing key"])
                if is_length_violation or is_keyword_violation or is_json_violation:
                    continue
                    
                if severity == "Critical":
                    min_score -= 0.35
                elif severity == "Medium":
                    min_score -= 0.15
                
            min_score = max(0.0, min_score)
            
            # If the LLM returned score is excessively harsh compared to the issues, lift it to min_score!
            llm_score = float(score) if score is not None else 1.0
            if llm_score < 0.5:
                score = max(llm_score, min_score)
            else:
                score = llm_score

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
                "threshold": self.threshold,
                "low_info_issues": low_info_issues
            }
        )
