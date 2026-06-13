from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """Abstract base class representing an LLM-powered Agent interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generates a text response from the agent given a prompt.

        Args:
            prompt (str): The prompt query string.

        Returns:
            str: The generated response text.
            
        Raises:
            Exception: If generation fails due to API errors, timeouts, or policy blocks.
        """
        pass
