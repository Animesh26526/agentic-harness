import os
import requests
import json
import time
from typing import Any, Dict, Optional
from google import genai
from google.genai.errors import APIError

class ModelProviderError(Exception):
    """Base exception for provider errors."""
    pass

class ModelProvider:
    """
    Generic provider abstraction to routing prompts to Google GenAI,
    Groq, or OpenRouter based on model name and environment configuration.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.gemini_key = api_key or os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")

        # Initialize Google GenAI client if key is present
        self.client = None
        if self.gemini_key:
            self.client = genai.Client(api_key=self.gemini_key)

    def generate(self, model_name: str, prompt: str) -> str:
        """
        Generates a response from the selected model using the appropriate provider.
        """
        # Map user-friendly names to internal API identifiers
        model_id = self._map_model_id(model_name)

        # 1. Routing for Llama 3.1 8B Instant
        if "llama" in model_id.lower():
            # Try Groq Provider
            if self.groq_key:
                try:
                    return self._generate_groq("llama-3.1-8b-instant", prompt)
                except Exception as e:
                    raise ModelProviderError(f"Groq API call failed: {str(e)}")
            
            # Try OpenRouter Provider
            if self.openrouter_key:
                try:
                    return self._generate_openrouter("meta-llama/llama-3.1-8b-instruct", prompt)
                except Exception as e:
                    raise ModelProviderError(f"OpenRouter API call failed: {str(e)}")

            # Try OpenAI compatible endpoint if set
            if self.openai_key:
                try:
                    return self._generate_openai("meta-llama/llama-3.1-8b-instruct", prompt)
                except Exception as e:
                    raise ModelProviderError(f"OpenAI API call failed: {str(e)}")

            # No keys found, raise a clear error (not a silent fallback)
            raise ModelProviderError(
                "Llama 3.1 8B Instant requires GROQ_API_KEY or OPENROUTER_API_KEY "
                "to be configured in your environment or .env file."
            )

        # 2. Routing for Gemma models
        elif "gemma" in model_id.lower():
            # Gemma models run through Gemini API
            # Make sure it has 'models/' prefix
            gemma_model = model_id
            if not gemma_model.startswith("models/"):
                gemma_model = f"models/{gemma_model}"
            return self._generate_gemini(gemma_model, prompt)

        # 3. Routing for Gemini models
        else:
            return self._generate_gemini(model_id, prompt)

    def _map_model_id(self, model_name: str) -> str:
        """Maps user-facing model selection strings to API model IDs."""
        mapping = {
            "Gemma 4 26B": "gemma-4-26b-a4b-it",
            "Gemma 4 31B": "gemma-4-31b-it",
            "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
            "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
            "gemma-4-31b-it": "gemma-4-31b-it",
            "models/gemma-4-26b-a4b-it": "models/gemma-4-26b-a4b-it",
            "models/gemma-4-31b-it": "models/gemma-4-31b-it"
        }
        # Default fallback to the name itself or gemini-2.5-flash-lite
        val = mapping.get(model_name, model_name)
        if not val:
            return "gemini-2.5-flash-lite"
        return val

    def _generate_gemini(self, model_id: str, prompt: str) -> str:
        """Call Gemini API using google-genai SDK client."""
        if not self.client:
            raise ModelProviderError("Gemini API client not initialized. GEMINI_API_KEY is missing.")
        
        # Enforce pacing delay dynamically
        try:
            min_spacing = float(os.getenv("GEMINI_PACING_DELAY", "1.0"))
        except Exception:
            min_spacing = 1.0

        last_request_time = float(os.getenv("GEMINI_LAST_REQUEST_TIME", "0.0"))
        elapsed = time.time() - last_request_time
        if elapsed < min_spacing:
            time.sleep(min_spacing - elapsed)

        os.environ["GEMINI_LAST_REQUEST_TIME"] = str(time.time())

        try:
            response = self.client.models.generate_content(
                model=model_id,
                contents=prompt
            )
            if not response or not response.text:
                raise ModelProviderError("Empty response returned from Gemini client.")
            return response.text.strip()
        except Exception as e:
            raise ModelProviderError(f"Gemini generation error: {str(e)}")

    def _generate_groq(self, model_id: str, prompt: str) -> str:
        """Call Groq API using requests."""
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=30)
        res.raise_for_status()
        res_json = res.json()
        return res_json["choices"][0]["message"]["content"].strip()

    def _generate_openrouter(self, model_id: str, prompt: str) -> str:
        """Call OpenRouter API using requests."""
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers, timeout=30)
        res.raise_for_status()
        res_json = res.json()
        return res_json["choices"][0]["message"]["content"].strip()

    def _generate_openai(self, model_id: str, prompt: str) -> str:
        """Call standard OpenAI endpoint using requests."""
        headers = {
            "Authorization": f"Bearer {self.openai_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}]
        }
        res = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers, timeout=30)
        res.raise_for_status()
        res_json = res.json()
        return res_json["choices"][0]["message"]["content"].strip()
