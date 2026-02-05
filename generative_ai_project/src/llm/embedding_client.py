"""
Client per modelli embedding del cluster UniBS.
Supporta: qwen3-embedding, nomic-embed-text-v1.5
"""
import os
import logging
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("EmbeddingClient")

class UniBSEmbeddingClient:
    """Client per generare embeddings con il cluster UniBS."""
    
    def __init__(self, model: str = "nomic-embed-text-v1.5"):
        self.base_url = os.getenv("GPUSTACK_BASE_URL", "https://gpustack.ing.unibs.it/v1")
        self.api_key = os.getenv("GPUSTACK_API_KEY")
        
        if not self.api_key:
            raise ValueError("GPUSTACK_API_KEY non configurata")
        
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = model
        
        # Validazione modello
        valid_models = ["qwen3-embedding", "nomic-embed-text-v1.5"]
        if model not in valid_models:
            logger.warning(f"Modello {model} non in lista ufficiale: {valid_models}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings per una lista di testi.
        
        Args:
            texts: Lista di stringhe da embedded
        
        Returns:
            Lista di vettori (liste di float)
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.info(f"✅ Generati {len(embeddings)} embeddings (dim={len(embeddings[0])})")
            return embeddings
        
        except Exception as e:
            logger.error(f"❌ Errore embedding: {e}")
            raise
