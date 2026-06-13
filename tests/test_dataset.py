import os
import json
import pytest

def test_dataset_completeness():
    """Verify that the benchmark dataset contains exactly 40 samples."""
    dataset_path = "data/benchmark_dataset.json"
    assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    assert len(dataset) == 40, f"Expected 40 samples, found {len(dataset)}"

def test_dataset_category_balance():
    """Verify that each category contains exactly 10 samples."""
    dataset_path = "data/benchmark_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    categories = [sample["category"] for sample in dataset]
    from collections import Counter
    counts = Counter(categories)
    
    expected_categories = {"structured_json", "constraint_following", "factual_qa", "extraction_math"}
    assert set(counts.keys()) == expected_categories, f"Invalid categories in dataset: {counts.keys()}"
    
    for cat in expected_categories:
        assert counts[cat] == 10, f"Expected 10 samples for '{cat}', found {counts[cat]}"

def test_dataset_unique_query_ids():
    """Verify that all query_id values are unique."""
    dataset_path = "data/benchmark_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    query_ids = [sample["query_id"] for sample in dataset]
    assert len(query_ids) == len(set(query_ids)), "Duplicate query_ids found in dataset!"

def test_dataset_required_fields():
    """Verify that every sample contains all required fields with appropriate structure."""
    dataset_path = "data/benchmark_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
        
    for idx, sample in enumerate(dataset):
        assert "query_id" in sample, f"Sample at index {idx} missing 'query_id'"
        assert "category" in sample, f"Sample at index {idx} missing 'category'"
        assert "input" in sample, f"Sample at index {idx} missing 'input'"
        assert "expected_output" in sample, f"Sample at index {idx} missing 'expected_output'"
        assert "evaluation_config" in sample, f"Sample at index {idx} missing 'evaluation_config'"
        
        config = sample["evaluation_config"]
        category = sample["category"]
        
        if category == "structured_json":
            assert config.get("validate_json") is True, f"JSON sample '{sample['query_id']}' missing 'validate_json: true'"
            assert "required_fields" in config, f"JSON sample '{sample['query_id']}' missing 'required_fields'"
            assert "field_types" in config, f"JSON sample '{sample['query_id']}' missing 'field_types'"
            
        elif category == "constraint_following":
            assert "forbidden_keywords" in config, f"Constraint sample '{sample['query_id']}' missing 'forbidden_keywords'"
            assert "max_length" in config, f"Constraint sample '{sample['query_id']}' missing 'max_length'"
            
        elif category == "factual_qa":
            assert "reference_text" in config, f"QA sample '{sample['query_id']}' missing 'reference_text'"
            
        elif category == "extraction_math":
            assert "reference_text" in config, f"Math sample '{sample['query_id']}' missing 'reference_text'"
