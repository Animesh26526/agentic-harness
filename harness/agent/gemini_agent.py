import time
from google import genai
from google.genai.errors import APIError
from harness.config import Config
from harness.agent.base_agent import BaseAgent

# Custom Exception definitions for robust error handling
class GeminiError(Exception):
    """Base exception class for all errors related to the Gemini Agent."""
    pass

class RateLimitError(GeminiError):
    """Raised when the Gemini API rate limit (429 ResourceExhausted) is hit."""
    pass

class EmptyResponseError(GeminiError):
    """Raised when the Gemini API returns a response containing no text or candidates."""
    pass

class SafetyBlockError(GeminiError):
    """Raised when Gemini blocks the prompt or response candidates due to safety policies."""
    pass

class GeminiGenerationError(GeminiError):
    """Raised when the model generation fails due to network, API parameters, or unexpected issues."""
    pass


class GeminiAgent(BaseAgent):
    """Production-friendly wrapper around Google's Gemini API with strict exceptions."""

    # Class-level variable to enforce minimum request pacing across all instances
    _last_request_time: float = 0.0

    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initializes the Gemini client.

        Args:
            api_key (str, optional): The Gemini API key. Defaults to loading from config.
            model_name (str, optional): The Gemini model name. Defaults to config default.
        """
        import os
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or Config.GEMINI_API_KEY
        resolved_model = (
            model_name 
            or os.getenv("DEFAULT_MODEL") 
            or Config.DEFAULT_MODEL 
            or os.getenv("DEFAULT_GEMINI_MODEL") 
            or Config.DEFAULT_GEMINI_MODEL
        )

        model_lower = str(resolved_model).lower()
        if "llama" in model_lower:
            self.model_name = "llama-3.1-8b-instant"
        elif "gemma-4-26b" in model_lower or "gemma 4 26b" in model_lower:
            self.model_name = "models/gemma-4-26b-a4b-it"
        elif "gemma-4-31b" in model_lower or "gemma 4 31b" in model_lower:
            self.model_name = "models/gemma-4-31b-it"
        elif "gemini-2.5-flash-lite" in model_lower or "gemini 2.5 flash lite" in model_lower:
            self.model_name = "gemini-2.5-flash-lite"
        elif "gemini-2.5-flash" in model_lower or "gemini 2.5 flash" in model_lower:
            self.model_name = "gemini-2.5-flash"
        elif "gemini-2.5-pro" in model_lower or "gemini 2.5 pro" in model_lower:
            self.model_name = "gemini-2.5-pro"
        else:
            self.model_name = resolved_model

        if not self.api_key and not os.getenv("GROQ_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError(
                "Gemini API key must be provided or configured via the GEMINI_API_KEY environment variable."
            )

        if self.model_name.startswith("gemma-") and not self.model_name.startswith("models/"):
            self.model_name = f"models/{self.model_name}"

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate(self, prompt: str) -> str:
        """
        Generates a text response using Gemini.
        Incorporates global request pacing and exponential backoff retries.

        Args:
            prompt (str): The prompt query string.

        Returns:
            str: The generated response text, or empty string on empty input.
            
        Raises:
            SafetyBlockError: If safety filters block content.
            EmptyResponseError: If response contains no generated content.
            RateLimitError: If API limits are exceeded.
            GeminiGenerationError: For other general execution exceptions.
        """
        if not prompt or not prompt.strip():
            return ""

        # Delegate Llama 3.1 8B Instant or requests when Gemini client is not initialized
        if "llama" in self.model_name.lower() or not self.client:
            from harness.agent.providers import ModelProvider
            provider = ModelProvider(api_key=self.api_key)
            try:
                return provider.generate(self.model_name, prompt)
            except Exception as e:
                raise GeminiGenerationError(f"Model generation failed: {str(e)}")

        backoffs = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        max_attempts = len(backoffs) + 1
        attempt = 1

        while True:
            # Enforce centralized request pacing (minimum spacing between any two calls)
            current_time = time.time()
            
            # Read last request time from global environment to be immune to duplicate class/module imports
            import os
            last_request_time_str = os.getenv("GEMINI_LAST_REQUEST_TIME", "0.0")
            try:
                last_request_time = float(last_request_time_str)
            except (ValueError, TypeError):
                last_request_time = 0.0

            elapsed = current_time - last_request_time
            
            # Read pacing delay dynamically from environment to handle runtime updates (e.g. from UI sliders)
            try:
                min_spacing = float(os.getenv("GEMINI_PACING_DELAY", "1.0"))
            except (ValueError, TypeError):
                min_spacing = 1.0

            if elapsed < min_spacing:
                time.sleep(min_spacing - elapsed)
                current_time = time.time()

            try:
                # Update timestamp globally right before the actual API call
                os.environ["GEMINI_LAST_REQUEST_TIME"] = str(current_time)
                GeminiAgent._last_request_time = current_time
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                
                # Handle prompt blocks or empty responses due to safety blocks
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    raise SafetyBlockError(
                        f"Request was blocked by Gemini safety filters. Reason: {response.prompt_feedback.block_reason}"
                    )

                if not response or not response.candidates:
                    raise EmptyResponseError("Empty response returned from model candidate generation.")

                # Check candidate finish reason
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason
                    if finish_reason and str(finish_reason).upper() == "SAFETY":
                        raise SafetyBlockError(
                            "Request was blocked by Gemini safety filters. Reason: SAFETY"
                        )

                # Retrieve text safely as response.text raises exception if candidates were blocked mid-stream
                try:
                    text = response.text
                except Exception as e:
                    # Check safety block in case text lookup raises exception
                    is_safety = False
                    if response.candidates:
                        candidate = response.candidates[0]
                        finish_reason = candidate.finish_reason
                        if finish_reason and str(finish_reason).upper() == "SAFETY":
                            is_safety = True
                    if is_safety:
                        raise SafetyBlockError(
                            "Request was blocked by Gemini safety filters. Reason: SAFETY"
                        )
                    raise EmptyResponseError(f"Model response text was unavailable: {str(e)}")

                if not text or not text.strip():
                    raise EmptyResponseError("Model response text was empty.")
                return text.strip()

            except (SafetyBlockError, EmptyResponseError) as e:
                # Re-raise direct exceptions
                raise e
            except Exception as e:
                error_msg = str(e)
                
                # Inspect for potential safety-blocked text exceptions that trigger on text lookup
                is_safety = False
                try:
                    if 'response' in locals() and response.prompt_feedback and response.prompt_feedback.block_reason:
                        is_safety = True
                except Exception:
                    pass
                try:
                    if 'response' in locals() and response.candidates:
                        candidate = response.candidates[0]
                        finish_reason = candidate.finish_reason
                        if finish_reason and str(finish_reason).upper() == "SAFETY":
                            is_safety = True
                except Exception:
                    pass
                    
                if is_safety:
                    raise SafetyBlockError(
                        f"Request was blocked by Gemini safety filters."
                    )

                # Handle rate-limit and transient specific error indications (429, 500, 502, 503, 504)
                is_transient = False
                is_429 = False
                if isinstance(e, APIError):
                    is_429 = (e.code == 429) or any(
                        token in error_msg for token in ("429", "ResourceExhausted")
                    )
                    is_transient = is_429 or (e.code in (500, 502, 503, 504)) or any(
                        token in error_msg for token in ("500", "502", "503", "504", "UNAVAILABLE", "SERVICE_UNAVAILABLE")
                    )
                else:
                    is_429 = any(
                        token in error_msg for token in ("429", "ResourceExhausted")
                    )
                    is_transient = is_429 or any(
                        token in error_msg for token in ("500", "502", "503", "504", "UNAVAILABLE", "SERVICE_UNAVAILABLE")
                    )
                
                if is_transient and attempt < max_attempts:
                    sleep_time = None
                    
                    # 1. Try to extract retry delay from error message (e.g. "Please retry in 50.26s")
                    import re
                    match = re.search(r"Please retry in ([0-9.]+)s", error_msg, re.IGNORECASE)
                    if match:
                        try:
                            sleep_time = float(match.group(1)) + 2.0
                        except Exception:
                            pass
                            
                    # 2. Try to extract retry delay from APIError details dict
                    if sleep_time is None:
                        try:
                            if isinstance(e, APIError) and isinstance(e.details, dict):
                                details_list = e.details.get('error', {}).get('details', [])
                                for detail in details_list:
                                    if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                        delay_str = detail.get('retryDelay', '')
                                        if delay_str.endswith('s'):
                                            sleep_time = float(delay_str[:-1]) + 1.5
                        except Exception:
                            pass

                    # 3. Fallback logic: if it's a 429 rate limit error, sleep longer than a general transient error
                    if sleep_time is None:
                        if is_429:
                            # 429 rate limit requires much longer backoff to reset quota window
                            sleep_time = max(15.0, backoffs[attempt - 1] * 2.0)
                        else:
                            sleep_time = backoffs[attempt - 1]

                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                
                if is_transient:
                    if is_429:
                        raise RateLimitError(
                            f"Gemini API rate limit exceeded (429 ResourceExhausted) after {max_attempts} attempts: {error_msg}"
                        )
                    else:
                        raise GeminiGenerationError(
                            f"Gemini API transient failure ({getattr(e, 'code', '503')} UNAVAILABLE) after {max_attempts} attempts: {error_msg}"
                        )
                    
                raise GeminiGenerationError(f"Gemini generation failed: {error_msg}")
