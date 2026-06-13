import pytest
from harness.evaluation_router import get_evaluators

def test_router_valid_categories():
    """Verify that valid categories return their correct designated evaluators."""
    assert get_evaluators("structured_json") == ["rule"]
    assert get_evaluators("constraint_following") == ["rule", "critic"]
    assert get_evaluators("factual_qa") == ["semantic", "critic"]
    assert get_evaluators("extraction_math") == ["semantic", "rule"]

def test_router_invalid_categories():
    """Verify that invalid categories raise a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        get_evaluators("invalid_category")
    assert "Invalid category" in str(excinfo.value)

    with pytest.raises(ValueError):
        get_evaluators("")

    with pytest.raises(ValueError):
        get_evaluators(None)  # type: ignore
