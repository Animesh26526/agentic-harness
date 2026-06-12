import os
from dataclasses import dataclass, field
from typing import Set
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# Frozen set of central task categories. Do NOT modify or rename these categories.
# Future orchestrator routing depends on these exact strings.
TASK_CATEGORIES: Set[str] = {
    "structured_json",
    "constraint_following",
    "factual_qa",
    "extraction_math"
}

@dataclass(frozen=True)
class AppConfig:
    """Centralized read-only configuration class for the Agentic Harness framework."""
    
    # API Credentials & Setup
    GEMINI_API_KEY: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    GROQ_API_KEY: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    DEFAULT_GEMINI_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_GEMINI_MODEL", "Llama 3.1 8B Instant")
    )
    MODEL_PROVIDER: str = field(
        default_factory=lambda: os.getenv("MODEL_PROVIDER", "google")
    )
    DEFAULT_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-lite")
    )
    
    # Environment Setup
    ENV: str = field(
        default_factory=lambda: os.getenv("ENV", "development")
    )
    
    # Evaluator Constraints & Thresholds
    SEMANTIC_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("SEMANTIC_THRESHOLD", "0.75"))
    )
    CRITIC_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("CRITIC_THRESHOLD", "0.80"))
    )
    RELIABILITY_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("RELIABILITY_THRESHOLD", "0.80"))
    )
    MAX_RESPONSE_LENGTH: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESPONSE_LENGTH", "2000"))
    )
    MAX_RETRIES: int = field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "2"))
    )
    
    # Model Caching Location for Local Deployment
    MODEL_CACHE_FOLDER: str = field(
        default_factory=lambda: os.getenv("MODEL_CACHE_FOLDER", "./models")
    )

    def validate(self) -> None:
        """
        Validates that required credentials are set.
        
        Raises:
            ValueError: If mandatory parameters are missing.
        """
        if not self.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY "
                "environment variable or configure it in your .env file."
            )
        if not self.DEFAULT_GEMINI_MODEL:
            raise ValueError(
                "DEFAULT_GEMINI_MODEL is not set or is empty."
            )
        if self.MODEL_CACHE_FOLDER:
            os.makedirs(self.MODEL_CACHE_FOLDER, exist_ok=True)

# Instantiated configuration object for global use
Config = AppConfig()

