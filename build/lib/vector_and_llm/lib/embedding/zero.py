#!/usr/bin/env python3
"""
Zero vector embedding backend (for testing without embeddings).
"""

from typing import List
from .base import EmbeddingBackend


class ZeroEmbeddingBackend(EmbeddingBackend):
    """Zero vector backend (no real embeddings)."""
    
    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size
    
    def get_embedding(self, text: str) -> List[float]:
        """Return a zero vector of the specified size."""
        return [0.0] * self.vector_size