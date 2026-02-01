import os
import time
import logging
import yaml
from typing import Dict, List, Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

# Load environment variables from .env file (if present)
load_dotenv()

logger = logging.getLogger("AI_Client")

class OpenAIClient:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize OpenAI client using Environment Variables for security.
        Configuration for model parameters is loaded from YAML.
        """
        
        # 1. Load non-sensitive configuration (params, paths) from YAML
        self.config = {}
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}. Using internal defaults.")

        # 2. Retrieve Credentials from Environment Variables (Security Best Practice)
        # We do NOT read keys from the YAML file anymore.
        self.api_key = os.getenv("GPUSTACK_API_KEY")
        self.base_url = os.getenv("GPUSTACK_BASE_URL", "https://gpustack.ing.unibs.it/v1")
        
        # 3. Validation
        if not self.api_key:
            logger.critical("❌ CRITICAL: API Key not found! Please create a .env file with GPUSTACK_API_KEY.")
            raise ValueError("Missing API Key in environment variables.")

        # 4. Initialize OpenAI Client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        # 5. Set Default Parameters
        # Priority: YAML config -> Defaults
        openai_cfg = self.config.get('openai', {})
        self.model_name = openai_cfg.get('model_name', "phi4-mini")
        
        self.llm_params = self.config.get('llm_parameters', {
            'temperature': 0.8,
            'top_p': 0.95,
            'max_tokens': 1024,
            'frequency_penalty': 0,
            'presence_penalty': 0
        })

    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        model: str = None,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Send messages to the model with automatic retry logic for stability.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Override default model (optional)
            max_retries: Number of retry attempts for temporary errors
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Model response content
        """
        target_model = model or self.model_name
        
        # Merge default params with runtime kwargs
        params = {**self.llm_params, **kwargs}
        params['model'] = target_model
        params['messages'] = messages

        for attempt in range(max_retries):
            try:
                # API Call
                response = self.client.chat.completions.create(**params)
                
                # Success
                return response.choices[0].message.content or ""

            except (APITimeoutError, RateLimitError) as e:
                # Handle temporary network/server issues
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                logger.warning(f"⚠️ AI Temporary Error (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            
            except APIError as e:
                # Handle fatal API errors (e.g., invalid request)
                logger.error(f"❌ AI API Error: {e}")
                break
            
            except Exception as e:
                # Handle unexpected Python errors
                logger.error(f"❌ Unexpected Error during generation: {e}")
                break
                
        logger.error("Failed to generate response after max retries.")
        return ""
