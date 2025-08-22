#!/usr/bin/env python3
"""
Base embedding backend interface.
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingBackend(ABC):
    """Base class for embedding backends."""
    
    def __init__(self):
        self.vector_size = 384  # Default, overridden by implementations
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def get_vector_size(self) -> int:
        """Get the vector size for this embedding model."""
        return self.vector_size