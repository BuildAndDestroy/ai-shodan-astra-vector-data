#!/usr/bin/env python3
"""
Base LLM backend interface.
"""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Base class for LLM backends."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response given prompt and context. Must be implemented by subclasses."""
        raise NotImplementedError