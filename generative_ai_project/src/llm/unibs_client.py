import os
import time
import logging
import yaml
from typing import Dict, List, Optional
from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("UniBS_Client")

class UniBSLLMClient:
    def __init__(self, config_path: str = "config/model_config.yaml", model_override: str = None):
        
        # 1. Load YAML Config
        self.config = {}
        try:
            with open(config_path, "r", encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config not found at {config_path}")

        # 2. Setup Credentials
        self.api_key = os.getenv("GPUSTACK_API_KEY")
        self.base_url = (
            os.getenv("GPUSTACK_BASE_URL") or 
            self.config.get('unibs_cluster', {}).get('base_url') or 
            "https://gpustack.ing.unibs.it/v1"
        )

        if not self.api_key:
            # Fallback per debug locale se non c'è .env (opzionale)
            logger.warning("No API Key in env. Checking if code runs without auth (unlikely).")

        # 3. Setup Client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # 4. Determine Model
        self.model_name = (
            model_override or 
            self.config.get('active_model') or 
            self.config.get('openai', {}).get('model_name') or # Retrocompatibilità
            "phi4-mini"
        )
        
        self.llm_params = self.config.get('llm_parameters', {
            'temperature': 0.8, 'max_tokens': 1024
        })
        
        logger.info(f"🤖 LLM Client initialized. Target: {self.model_name}")

        params = {**self.llm_params, **kwargs}
        params['model'] = self.model_name
        params['messages'] = messages
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content or ""
            except (APITimeoutError, RateLimitError) as e:
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error: {e}")
                break
        return ""

def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, str]:
    """
    Ritorna sia content che reasoning_content se disponibile.
    
    Returns:
        dict: {'content': str, 'reasoning': str|None}
    """
    params = {**self.llm_params, **kwargs}
    params['model'] = self.model_name
    params['messages'] = messages
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = self.client.chat.completions.create(**params)
            choice = response.choices[0]
            
            return {
                'content': choice.message.content or "",
                'reasoning': getattr(choice.message, 'reasoning_content', None)
            }
        except (APITimeoutError, RateLimitError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.error(f"Error: {e}")
            break
    
    return {'content': "", 'reasoning': None}
