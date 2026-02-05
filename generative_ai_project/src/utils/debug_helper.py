"""
Utility di debug per testare connessione al cluster UniBS.
Basato su Sezione 8 del vademecum.
"""
import os
import subprocess
import json
from pathlib import Path

def test_cluster_connection():
    """Testa connessione raw con cURL come da vademecum."""
    api_key = os.getenv("GPUSTACK_API_KEY")
    
    if not api_key:
        print("❌ GPUSTACK_API_KEY non trovata!")
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
    
    print("🔧 Test connessione cluster UniBS...")
    print(f"🔗 Endpoint: https://gpustack.ing.unibs.it/v1")
    print(f"🔑 API Key: {api_key[:10]}...")
    
    try:
        result = subprocess.run(
            curl_command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Connessione OK!")
            print(f"📥 Risposta: {result.stdout[:200]}")
            return True
        else:
            print(f"❌ Errore: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"❌ Eccezione: {e}")
        return False

if __name__ == "__main__":
    test_cluster_connection()
