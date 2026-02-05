"""
UniBS LLM Cluster Client
OpenAI-compatible client for the University of Brescia GPU cluster.
Compliant with UniBS Cluster Handbook (February 2026).

Author: Brilant Gashi
"""

import os
import yaml
import logging
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class UniBSLLMClient:
    """
    Client for interacting with UniBS LLM cluster via OpenAI-compatible API.
    
    Features:
    - Automatic API key loading from .env
    - Model override support for ensemble processing
    - Reasoning content extraction (for supported models)
    - Robust error handling
    """
    
    def __init__(self, config_path: str, model_override: str = None, **kwargs):
        """
        Initialize UniBS LLM client.
        
        Args:
            config_path: Path to model_config.yaml
            model_override: Optional model name to override config
            **kwargs: Additional parameters for OpenAI client
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Get credentials from environment
        self.api_key = os.getenv('GPUSTACK_API_KEY')
        self.base_url = os.getenv('GPUSTACK_BASE_URL', 
                                   'https://gpustack.ing.unibs.it/v1')
        
        if not self.api_key:
            logger.warning("⚠️  No API Key in env. Checking if code runs without auth (unlikely).")
        
        # Initialize OpenAI client
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        
        # Set model (override or from config)
        self.model = model_override or self.config.get('active_model')
        
        # Load LLM parameters from config
        self.llm_params = self.config.get('llm_params', {})
        
        # Merge with any additional kwargs
        self.llm_params = {**self.llm_params, **kwargs}
        
        logger.info(f"✅ UniBSLLMClient initialized | Model: {self.model} | Base URL: {self.base_url}")
    
    def generate_response(self, messages: list, **override_params) -> dict:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **override_params: Optional parameters to override config
        
        Returns:
            dict: Response with 'content' and optional 'reasoning' keys
        """
        # Merge parameters: config < instance < call-time
        params = {
            **self.llm_params,
            **override_params,
            'model': self.model,
            'messages': messages
        }
        
        try:
            response = self.client.chat.completions.create(**params)
            
            # Extract content
            content = response.choices[0].message.content
            
            # Extract reasoning if available (from extended_thinking models)
            reasoning = None
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning = response.choices[0].message.reasoning_content
            
            result = {
                'content': content,
                'reasoning': reasoning,
                'model': response.model,
                'usage': response.usage.model_dump() if response.usage else None
            }
            
            logger.debug(f"✅ Response generated | Model: {result['model']} | Tokens: {result.get('usage', {}).get('total_tokens', 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection to UniBS cluster.
        
        Returns:
            bool: True if connection successful
        """
        try:
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'test' if you can hear me."}
            ]
            
            response = self.generate_response(test_messages)
            
            if response.get('content'):
                logger.info(f"✅ Connection test successful! Model responded: {response['content'][:50]}...")
                return True
            
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_available_models(config_path: str) -> list:
    """
    Get list of available models from config.
    
    Args:
        config_path: Path to model_config.yaml
    
    Returns:
        list: Available model names
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    ensemble = config.get('ensemble_models', [])
    active = config.get('active_model')
    
    models = list(set(ensemble + [active])) if ensemble else [active]
    return [m for m in models if m]  # Remove None values


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation (4 chars ≈ 1 token).
    
    Args:
        text: Input text
    
    Returns:
        int: Estimated token count
    """
    return len(text) // 4


# ==============================================================================
# MAIN BLOCK (for standalone testing)
# ==============================================================================

if __name__ == "__main__":
    # Setup logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test configuration
    config_path = Path(__file__).parent.parent.parent / "config" / "model_config.yaml"
    
    print("\n" + "="*70)
    print("🧪  UniBS LLM Client - Connection Test")
    print("="*70)
    
    try:
        # Initialize client
        client = UniBSLLMClient(config_path=str(config_path))
        
        # Test connection
        if client.test_connection():
            print("✅ Client is working correctly!")
            
            # Get available models
            models = get_available_models(str(config_path))
            print(f"\n📋 Available models: {', '.join(models)}")
        else:
            print("❌ Connection test failed. Check your .env configuration.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("="*70 + "\n")
