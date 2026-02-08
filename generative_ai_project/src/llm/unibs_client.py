"""
UniBS LLM Client - OpenAI SDK Implementation
Compliant with UniBS Cluster Handbook (February 2026)
"""

import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Setup logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class UniBSLLMClient:
    """
    Client for UniBS LLM cluster using official OpenAI SDK.
    
    Reference: UniBS Cluster Handbook, Section 4
    Endpoint: https://gpustack.ing.unibs.it/v1
    """
    
    def __init__(self, config_path: str, model_override: Optional[str] = None):
        """
        Initialize the UniBS LLM client with OpenAI SDK.
        
        Args:
            config_path: Path to model_config.yaml
            model_override: Optional model name to override config
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Model selection
        self.model_name = model_override or self.config.get('active_model', 'qwen3')
        
        # API configuration
        api_key = os.environ.get("GPUSTACK_API_KEY")
        if not api_key:
            raise ValueError(
                "GPUSTACK_API_KEY not found. "
                "Ensure .env file exists in project root with: GPUSTACK_API_KEY=your_key"
            )
        
        # Initialize OpenAI client with UniBS endpoint
        self.client = OpenAI(
            base_url="https://gpustack.ing.unibs.it/v1",
            api_key=api_key,
            timeout=480.0  # 8 minutes timeout
        )
        
        # Get LLM parameters from config
        self.llm_params = self.config.get('llm_parameters', {})
        
        # ✅ CHANGED: print() → logger.debug()
        logger.debug(f"UniBSLLMClient initialized | Model: {self.model_name}")
    
    def generate_response(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Generate response using OpenAI SDK.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Dict with 'content' and optional 'reasoning' keys
        
        Raises:
            Exception: If API call fails
        """
        try:
            # Call API using OpenAI SDK (handbook compliant)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.llm_params.get('temperature', 0.6),
                top_p=self.llm_params.get('top_p', 0.95),
                max_tokens=self.llm_params.get('max_tokens', 1024),
                frequency_penalty=self.llm_params.get('frequency_penalty', 0),
                presence_penalty=self.llm_params.get('presence_penalty', 0),
            )
            
            # Extract response (handbook Section 4)
            choice = response.choices[0]
            
            result = {
                'content': choice.message.content
            }
            
            # Add reasoning if available (handbook page 2)
            if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
                result['reasoning'] = choice.message.reasoning_content
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"API call failed for model {self.model_name}: {e}")


# Test when run directly
if __name__ == "__main__":
    print("="*70)
    print("🧪  UniBS LLM Client - Connection Test")
    print("="*70)
    
    try:
        # Check if API key is loaded
        api_key = os.environ.get("GPUSTACK_API_KEY")
        if api_key:
            print(f"✅ API Key loaded: {api_key[:20]}...{api_key[-5:]}")
        else:
            print("❌ API Key not found in environment")
            exit(1)
        
        # Initialize client
        client = UniBSLLMClient(
            config_path="config/model_config.yaml",
            model_override="phi4-mini"
        )
        
        # Test message
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, UniBS cluster!' in one sentence."}
        ]
        
        print("📤 Sending test request...")
        response = client.generate_response(test_messages)
        
        print("\n✅ Connection test successful!")
        print(f"📨 Response: {response['content']}")
        
        if 'reasoning' in response:
            print(f"🧠 Reasoning: {response['reasoning'][:100]}...")
        
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("="*70)
        exit(1)
