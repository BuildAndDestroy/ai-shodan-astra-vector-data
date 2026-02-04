#!/usr/bin/env python3
"""
Command-line tools for vector and LLM operations.
"""

# Tools are meant to be run as scripts, not imported
# But we can expose the main functions for programmatic use

from .shodan_to_qdrant import main as shodan_to_qdrant_main
from .llm_rag_shodan import main as llm_rag_shodan_main

__all__ = ['shodan_to_qdrant_main', 'llm_rag_shodan_main']