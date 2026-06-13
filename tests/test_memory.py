import os
import tempfile
import pytest
from harness.memory import MemoryManager, EvaluationMemory

@pytest.fixture
def temp_mem_db():
    """Fixture to create and clean up a temporary database file for memory testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    
    manager = MemoryManager(db_path=path)
    
    yield manager
    
    if os.path.exists(path):
        os.remove(path)

def test_store_and_retrieve_memory(temp_mem_db):
    """Test storing an evaluation run and retrieving it from memory."""
    prompt = "Translate Hello to French"
    response = "Bonjour"
    semantic_score = 0.95
    rule_score = 1.0
    critic_score = 0.9
    overall_score = 0.95
    issues = ["Formatting minor check"]
    corrections = ["Use capitalized greeting"]
    
    # Store memory
    inserted_id = temp_mem_db.store_evaluation(
        prompt=prompt,
        response=response,
        semantic_score=semantic_score,
        rule_score=rule_score,
        critic_score=critic_score,
        overall_score=overall_score,
        issues=issues,
        corrections=corrections
    )
    assert inserted_id > 0
    
    # Retrieve memories
    memories = temp_mem_db.retrieve_memories(limit=10)
    assert len(memories) == 1
    
    mem = memories[0]
    assert mem.id == inserted_id
    assert mem.prompt == prompt
    assert mem.response == response
    assert mem.semantic_score == semantic_score
    assert mem.rule_score == rule_score
    assert mem.critic_score == critic_score
    assert mem.overall_score == overall_score
    assert mem.issues == issues
    assert mem.corrections == corrections
    assert mem.timestamp is not None

def test_find_similar_memory(temp_mem_db):
    """Test finding a similar evaluation by exact/substring match."""
    temp_mem_db.store_evaluation(
        prompt="Write a Python script to sort a list",
        response="def sort_list(lst): ...",
        semantic_score=0.9,
        rule_score=1.0,
        critic_score=0.9,
        overall_score=0.93,
        issues=[],
        corrections=[]
    )
    
    # Search with exact match
    match = temp_mem_db.find_similar_memory("Write a Python script to sort a list")
    assert match is not None
    assert "def sort_list" in match.response
    
    # Search with similar query (which is a substring or container of the stored prompt)
    match_sub = temp_mem_db.find_similar_memory("Can you Write a Python script to sort a list right now?")
    assert match_sub is not None
    assert "def sort_list" in match_sub.response
    
    # Unrelated search
    match_none = temp_mem_db.find_similar_memory("How to bake a cake")
    assert match_none is None
