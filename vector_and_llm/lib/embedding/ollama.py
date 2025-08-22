#!/usr/bin/env python3
"""
Ollama embedding backend using nomic-embed-text.
"""

import requests
from typing import List
from .base import EmbeddingBackend


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend using nomic-embed-text."""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "nomic-embed-text"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.vector_size = 768  # nomic-embed-text dimensions
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API."""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            print(f"✗ Ollama embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * self.vector_size