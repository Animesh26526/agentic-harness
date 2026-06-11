#!/usr/bin/env python3
import os
import json
import re
import sys
from typing import Any, Dict, List, Tuple

# Ensure codebase packages can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.evaluators.rule_based import RuleBasedValidator

def analyze_case(case: Dict[str, Any]) -> Tuple[str, List[str]]:
    query_id = case.get("query_id", "unknown")
    category = case.get("category", "")
    prompt = case.get("input", "")
    expected_output = case.get("expected_output", "")
    config = case.get("evaluation_config", {})

    issues = []
    classification = "Valid"

    # ---------------------------------------------------------
    # 1. Evaluate Expected Output against Validator Constraints
    # ---------------------------------------------------------
    
    # 1a. Rule-based checks (max_length, forbidden_keywords, JSON schema, min_words)
    validator = RuleBasedValidator()
    # Evaluate expected output using the RuleBasedValidator logic
    eval_result = validator.evaluate(
        generated_text=expected_output,
        validate_json=config.get("validate_json", False),
        required_fields=config.get("required_fields"),
        field_types=config.get("field_types"),
        forbidden_keywords=config.get("forbidden_keywords"),
        max_length=config.get("max_length"),
        min_length=config.get("min_length"),
        min_words=config.get("min_words"),
        max_words=config.get("max_words")
    )

    if not eval_result.passed:
        # Expected output itself fails validator constraints
        issues.append(f"Expected output fails rule-based validation: {', '.join(eval_result.issues)}")
        classification = "Contradictory"

    # 1b. Semantic/Math check
    if category in ("factual_qa", "extraction_math") and "reference_text" in config:
        ref_text = config["reference_text"]
        # Basic check for exact substring or numeric equivalence
        clean_exp = expected_output.strip().lower()
        clean_ref = ref_text.strip().lower()
        
        is_numeric = clean_ref.replace('.', '', 1).isdigit()
        # If numeric and not equal, or if short fact is not matching
        if (len(clean_ref) <= 12 or is_numeric) and clean_ref not in clean_exp and clean_exp not in clean_ref:
            issues.append(f"Expected output '{expected_output}' does not semantically match reference '{ref_text}'")
            if classification != "Contradictory":
                classification = "Ambiguous"

    # ---------------------------------------------------------
    # 2. Compare Prompt Intent vs Validator Constraints
    # ---------------------------------------------------------

    # 2a. Character limit contradiction/ambiguity check
    # Regex to look for "under X characters", "max X characters", "exactly X characters", "limit of X"
    prompt_len_match = re.search(r"(?:under|exactly|limit of|max|maximum of|keep the answer under)\s*(\d+)\s*character", prompt, re.IGNORECASE)
    config_max_len = config.get("max_length")

    if prompt_len_match:
        prompt_len = int(prompt_len_match.group(1))
        # If prompt specifies a limit but config doesn't, or they mismatch
        if config_max_len is None:
            # Check if the prompt instructs ignoring the limit
            if not ("ignore" in prompt.lower() or "do not respect" in prompt.lower() or "do not follow" in prompt.lower()):
                issues.append(f"Prompt requests length limit of {prompt_len} but no max_length in validator config")
                if classification == "Valid":
                    classification = "Ambiguous"
        elif config_max_len != prompt_len:
            issues.append(f"Length limit mismatch: Prompt asks for {prompt_len} but config specifies {config_max_len}")
            classification = "Contradictory"

    # Check for "exactly X characters" in prompt
    prompt_exact_match = re.search(r"exactly\s*(\d+)\s*character", prompt, re.IGNORECASE)
    if prompt_exact_match:
        exact_len = int(prompt_exact_match.group(1))
        if config_max_len is not None:
            if exact_len > config_max_len:
                # E.g. prompt: "exactly 21 characters" but config: "max_length: 20"
                issues.append(f"Impossible constraint: Prompt asks for exactly {exact_len} characters, but validator limits max_length to {config_max_len}")
                classification = "Impossible"
            elif config.get("min_length") != exact_len:
                # E.g. prompt: "exactly 20 characters" and config: "max_length: 20"
                # This is Ambiguous/incomplete because any shorter response passes validator but violates exact prompt
                issues.append(f"Incomplete constraint matching: Prompt asks for exactly {exact_len} characters, but validator only checks max_length <= {config_max_len} (missing min_length)")
                if classification == "Valid":
                    classification = "Ambiguous"

    # Check for "do not respect" or "ignore" length/keyword limits
    ignore_limit_match = re.search(r"(?:do not respect|ignore|do not follow|do not keep to)\s*.*?limit", prompt, re.IGNORECASE)
    if ignore_limit_match and config_max_len is not None:
        issues.append(f"Contradictory instruction: Prompt requests to ignore limit, but validator config enforces max_length={config_max_len}")
        classification = "Contradictory"

    # 2b. Forbidden keywords contradiction/ambiguity check
    # Extract "forbidden words" mentioned in prompt
    # Example: "must not contain the words 'liquid', 'h2o'..."
    prompt_forbidden_match = re.findall(r"(?:not contain|not use|do not use|without using|forbidden)\s*[^.!?]*?['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    config_forbidden = config.get("forbidden_keywords", [])
    config_forbidden_lower = [k.lower() for k in config_forbidden]

    for word in prompt_forbidden_match:
        word_clean = word.lower().strip()
        # If prompt word is not in validator config
        if word_clean not in config_forbidden_lower:
            issues.append(f"Prompt forbids word '{word}' but it is not registered in validator config")
            if classification == "Valid":
                classification = "Ambiguous"

    # Check if config has forbidden keywords not in prompt
    for kw in config_forbidden:
        if kw == "```" and ("markdown" in prompt.lower() or "conversational" in prompt.lower() or "code block" in prompt.lower()):
            continue
        # Allow special characters described textually in prompt
        if kw == "\"" and ("double quote" in prompt.lower() or "quotes" in prompt.lower()):
            continue
        if kw == "}" and ("closing brace" in prompt.lower() or "curly brace" in prompt.lower() or "braces" in prompt.lower()):
            continue
        if kw == ":" and ("colon" in prompt.lower() or "colons" in prompt.lower()):
            continue
        if kw.lower() not in prompt.lower():
            issues.append(f"Validator config forbids '{kw}' but this is not mentioned in the prompt")
            if classification == "Valid":
                classification = "Ambiguous"

    # 2c. JSON Schema validation alignment check
    if config.get("validate_json", False):
        # Look for nested properties described in prompt but missing in required_fields / field_types
        # Example: location (object with city and zip)
        location_nested_match = re.search(r"location['\"]?\s*\(object\s*with\s*['\"]?city['\"]?\s*string\s*and\s*['\"]?zip['\"]?\s*integer\s*\)", prompt, re.IGNORECASE)
        field_types = config.get("field_types", {})
        if location_nested_match and "location" in field_types:
            if "location.city" not in field_types or "location.zip" not in field_types:
                issues.append("Nested JSON field types ('city' string, 'zip' integer) are described in prompt but not verified by validator config")
                if classification == "Valid":
                    classification = "Ambiguous"

        # Check for array items type matching (e.g., array of integers)
        array_ints_match = re.search(r"array\s*of\s*integers", prompt, re.IGNORECASE)
        if array_ints_match and "scores" in field_types:
            if field_types["scores"] == "array" or "scores" not in field_types:
                issues.append("Array element type ('integer') is described in prompt but not verified by validator config")
                if classification == "Valid":
                    classification = "Ambiguous"

        # Check if prompt explicitly asks for invalid JSON
        deliberate_invalid_match = re.search(r"(?:missing|deliberately missing|invalid json|no colons|no closing brace)", prompt, re.IGNORECASE)
        if deliberate_invalid_match and config.get("validate_json") != "invalid":
            issues.append("Contradictory prompt: asks for invalid/missing JSON structure, but validator enforces validate_json=True")
            classification = "Contradictory"

    # Check for negative formatting constraints (like banning markdown or conversational text) not verified by validator
    prompt_no_markdown_match = re.search(r"(?:do not include|do not output|no|without|do not write)\s*[^.!?]*?(?:markdown|conversational|code block|block|comments)", prompt, re.IGNORECASE)
    if prompt_no_markdown_match and "```" not in config_forbidden:
        issues.append("Negative formatting constraints (e.g. banning markdown/conversational text) are described in prompt but not verified by validator config (missing '```' in forbidden_keywords)")
        if classification == "Valid":
            classification = "Ambiguous"

    # 2d. Check word count limits
    prompt_words_match = re.search(r"(?:at least|minimum of)\s*(\d+)\s*word", prompt, re.IGNORECASE)
    config_min_words = config.get("min_words")
    if prompt_words_match:
        prompt_words = int(prompt_words_match.group(1))
        if config_min_words is None:
            issues.append(f"Prompt requests at least {prompt_words} words but no min_words in validator config")
            if classification == "Valid":
                classification = "Ambiguous"
        elif config_min_words != prompt_words:
            issues.append(f"Word count mismatch: Prompt asks for {prompt_words} but config specifies {config_min_words}")
            classification = "Contradictory"

    # Check for impossible combination of word/character limits
    if config_max_len is not None and config_min_words is not None:
        # Theoretical absolute minimum character length for N words is 2*N - 1 (e.g. "a b c d...")
        min_possible_len = 2 * config_min_words - 1
        if min_possible_len > config_max_len:
            issues.append(f"Impossible limits: min_words={config_min_words} requires at least {min_possible_len} characters, but max_length={config_max_len}")
            classification = "Impossible"
        elif min_possible_len == config_max_len:
            issues.append(f"Highly contradictory/restrictive limits: min_words={config_min_words} leaves no room for natural language under max_length={config_max_len}")
            classification = "Contradictory"

    return classification, issues

def run_sanity_check(filepath: str, output_clean_path: str = None) -> List[Dict[str, Any]]:
    print(f"\n=======================================================")
    print(f"Sanity Checking Dataset: {os.path.basename(filepath)}")
    print(f"=======================================================")
    
    with open(filepath, "r") as f:
        data = json.load(f)

    clean_data = []
    classifications = {"Valid": 0, "Ambiguous": 0, "Contradictory": 0, "Impossible": 0}
    
    for case in data:
        q_id = case.get("query_id")
        desc = case.get("description", case.get("input")[:50] + "...")
        classification, issues = analyze_case(case)
        classifications[classification] += 1

        print(f"\n[{classification}] {q_id}: {desc}")
        if issues:
            print("  Issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  No issues found. Fully valid.")

        if classification == "Valid":
            clean_data.append(case)

    print(f"\nSummary of classifications for {os.path.basename(filepath)}:")
    for cls, count in classifications.items():
        print(f"  {cls}: {count}")

    if output_clean_path and clean_data:
        os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
        with open(output_clean_path, "w") as f:
            json.dump(clean_data, f, indent=2)
        print(f"\nSaved {len(clean_data)} valid cases to {output_clean_path}")

    return clean_data

if __name__ == "__main__":
    benchmark_path = "/workspaces/agentic-harness/data/benchmark_dataset.json"
    challenge_path = "/workspaces/agentic-harness/data/challenge_dataset.json"
    
    clean_bench_path = "/workspaces/agentic-harness/data/clean_benchmark_dataset.json"
    clean_challenge_path = "/workspaces/agentic-harness/data/clean_challenge_dataset.json"
    
    run_sanity_check(benchmark_path, clean_bench_path)
    run_sanity_check(challenge_path, clean_challenge_path)
