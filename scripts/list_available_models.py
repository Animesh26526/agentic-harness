#!/usr/bin/env python3
import os
import sys
from dotenv import load_dotenv

# Ensure we load environment variables from the root .env file
load_dotenv()

try:
    from google import genai
    from google.genai.errors import APIError
except ImportError:
    print("Error: The modern google-genai library is not installed.", file=sys.stderr)
    print("Please install it with: pip install google-genai", file=sys.stderr)
    sys.exit(1)

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
        print("Please configure it in your environment or .env file.", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Initialize client with key
        client = genai.Client(api_key=api_key)
        
        # Call list_models API
        models = client.models.list()
        
        print("Available Gemini Models:")
        for model in models:
            name = model.name
            if name.startswith("models/"):
                name = name[len("models/"):]
            print(f"- {name}")
            
    except APIError as e:
        print(f"API Error listing models: {e} (Code: {getattr(e, 'code', 'unknown')})", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error listing models: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
