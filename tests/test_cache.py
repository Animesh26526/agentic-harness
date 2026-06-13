import os
import tempfile
import pytest
from harness.cache import ResponseCacheManager

@pytest.fixture
def temp_cache_db():
    """Fixture to create and clean up a temporary database file for cache testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    manager = ResponseCacheManager(db_path=path)
    
    yield manager
    
    if os.path.exists(path):
        os.remove(path)

def test_cache_set_and_get(temp_cache_db):
    """Test setting, getting, and updating cache entries."""
    prompt = "Create a user profile"
    model = "Gemini 2.5 Flash"
    harness_enabled = True
    result_data = {
        "final_response": "{\"user\": \"bob\"}",
        "raw_response": "bob profile",
        "overall_score": 0.95,
        "retry_count": 1,
        "passed": True,
        "issues": []
    }
    
    # 1. Retrieve non-existent cache entry
    cached = temp_cache_db.get(prompt, model, harness_enabled)
    assert cached is None
    
    # 2. Set cache entry
    temp_cache_db.set(prompt, model, harness_enabled, result_data)
    
    # 3. Retrieve stored cache entry
    cached = temp_cache_db.get(prompt, model, harness_enabled)
    assert cached is not None
    assert cached["final_response"] == "{\"user\": \"bob\"}"
    assert cached["overall_score"] == 0.95
    assert cached["passed"] is True
    
    # 4. Check different harness mode returns None
    cached_disabled = temp_cache_db.get(prompt, model, False)
    assert cached_disabled is None
    
    # 5. Update cache entry
    updated_data = dict(result_data)
    updated_data["overall_score"] = 1.0
    temp_cache_db.set(prompt, model, harness_enabled, updated_data)
    
    cached_updated = temp_cache_db.get(prompt, model, harness_enabled)
    assert cached_updated is not None
    assert cached_updated["overall_score"] == 1.0

def test_cache_clear(temp_cache_db):
    """Test clearing all cache entries."""
    temp_cache_db.set("Prompt 1", "modelA", True, {"response": "A"})
    temp_cache_db.set("Prompt 2", "modelB", False, {"response": "B"})
    
    assert temp_cache_db.get("Prompt 1", "modelA", True) is not None
    assert temp_cache_db.get("Prompt 2", "modelB", False) is not None
    
    temp_cache_db.clear()
    
    assert temp_cache_db.get("Prompt 1", "modelA", True) is None
    assert temp_cache_db.get("Prompt 2", "modelB", False) is None
