"""
Debug utility for testing LLM API connection.
"""
import os
import subprocess
import json
from pathlib import Path


def test_cluster_connection():
    """Test raw connection with cURL."""
    api_key = os.getenv("GPUSTACK_API_KEY")
    
    if not api_key:
        print("GPUSTACK_API_KEY not found!")
        return False
    
    curl_command = [
        "curl",
        "https://gpustack.ing.unibs.it/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", f"Authorization: Bearer {api_key}",
        "-d", json.dumps({
            "model": "qwen3",
            "messages": [
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Reply with 'OK' if you receive this."}
            ],
            "temperature": 0.6,
            "max_tokens": 50
        })
    ]
    
    print("Testing API connection...")
    print(f"Endpoint: https://gpustack.ing.unibs.it/v1")
    print(f"API Key: {api_key[:10]}...")
    
    try:
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("Connection OK!")
            print(f"Response: {result.stdout[:200]}")
            return True
        else:
            print(f"Error: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"Exception: {e}")
        return False


if __name__ == "__main__":
    test_cluster_connection()
