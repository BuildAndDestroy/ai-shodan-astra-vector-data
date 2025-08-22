#!/usr/bin/env python3
"""
Embedding backends for vector generation.
"""

from .base import EmbeddingBackend
from .zero import ZeroEmbeddingBackend
from .ollama import OllamaEmbeddingBackend
from .bedrock import BedrockEmbeddingBackend

__all__ = [
    'EmbeddingBackend',
    'ZeroEmbeddingBackend', 
    'OllamaEmbeddingBackend',
    'BedrockEmbeddingBackend'
]