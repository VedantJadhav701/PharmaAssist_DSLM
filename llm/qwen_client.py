import os
import logging
import requests
import json
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

logger = logging.getLogger(__name__)

class QwenClient:
    def __init__(self, host: str = None, model_name: str = None):
        # Default to environment variables or local fallback
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip('/')
        self.model_name = model_name or os.getenv("OLLAMA_MODEL") or "qwen2.5:1.5b-instruct-q4_K_M"
        logger.info(f"Initialized Ollama Client on {self.host} with model '{self.model_name}'")

    def generate_response(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        """
        Sends system and user prompts to the local Ollama instance and returns the generated text.
        Sets a very low temperature (default 0.1) to enforce strict medical facts and prevent hallucinations.
        """
        url = f"{self.host}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "options": {
                "temperature": temperature,
                "num_predict": 512, # limit max output size
                "top_p": 0.9,
                "seed": 42
            },
            "stream": False
        }
        
        try:
            logger.info(f"Sending generation request to local Ollama ({self.model_name})...")
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result_json = response.json()
            response_text = result_json.get("message", {}).get("content", "")
            return response_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling local Ollama service: {e}")
            raise RuntimeError(
                f"Failed to connect to local Ollama API at '{url}'. "
                "Ensure Ollama is running (`ollama serve` or Ollama App) "
                f"and that model '{self.model_name}' is installed (`ollama pull {self.model_name}`)."
            ) from e

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    client = QwenClient()
    try:
        ans = client.generate_response("You are a helpful assistant.", "Test message: say hello.")
        print("Response:", ans)
    except Exception as e:
        print("Error:", e)
