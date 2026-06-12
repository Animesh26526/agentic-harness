def compute_response_length(text: str) -> int:
    """
    Single source of truth for deterministic length validation.
    """
    if not text:
        return 0
    return len(text)
