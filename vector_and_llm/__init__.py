#!/usr/bin/env python3
"""
Vector and LLM package for Shodan data analysis with RAG.

This package provides tools and libraries for:
- Converting Shodan JSON data into vector embeddings
- Storing vectors in Qdrant database
- Performing RAG queries with various LLM backends

Main components:
- lib.embedding: Embedding backends (Ollama, AWS Bedrock, Zero)
- lib.llm: LLM backends for response generation
- lib.qdrant: Qdrant client and utilities
- tools: Command-line tools for import and querying
"""

__version__ = "1.0.0"
__author__ = "BuildAndDestroy"
__email__ = "info@example.com"

# Import main classes for convenience
from .lib.embedding import EmbeddingBackend, ZeroEmbeddingBackend, OllamaEmbeddingBackend, BedrockEmbeddingBackend
from .lib.llm import LLMBackend, OllamaLLMBackend, BedrockLLMBackend
from .lib.qdrant import QdrantClient

__all__ = [
    '__version__',
    '__author__', 
    '__email__',
    'EmbeddingBackend',
    'ZeroEmbeddingBackend',
    'OllamaEmbeddingBackend', 
    'BedrockEmbeddingBackend',
    'LLMBackend',
    'OllamaLLMBackend',
    'BedrockLLMBackend',
    'QdrantClient'
]