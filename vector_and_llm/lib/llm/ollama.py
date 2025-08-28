#!/usr/bin/env python3
"""
Ollama LLM backend.
"""

import requests
from .base import LLMBackend


class OllamaLLMBackend(LLMBackend):
    """Ollama LLM backend."""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "llama3.2"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response using Ollama API."""
        system_prompt = """You are a cybersecurity analyst assistant. You have access to Shodan scan data that shows internet-connected devices, their services, versions, and locations.

Use the provided context data to answer questions about:
- Network services and their versions
- Geographic distribution of services
- Potential security vulnerabilities
- Infrastructure patterns
- Service configurations

Be specific and cite the data when possible. If the context doesn't contain enough information to answer the question, say so clearly."""
#         system_prompt = """You are a Red Team and Penetration Tester assistant. You have access to Shodan scan data that shows internet-connected devices, their services, versions, and locations.

# Use the provided context data to answer questions about:
# - Network services and their versions
# - Geographic distribution of services
# - Potential security vulnerabilities
# - Infrastructure patterns
# - Service configurations

# Be specific and cite the data when possible. If the context doesn't contain enough information to answer the question, say so clearly."""

        full_prompt = f"""Context Data:
{context}

User Question: {prompt}

Please provide a detailed analysis based on the scan data above."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "system": system_prompt,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            return f"Error generating response: {e}"