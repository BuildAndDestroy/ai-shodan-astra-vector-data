#!/usr/bin/env python3
"""
AWS Bedrock LLM backend.
"""

import json
from .base import LLMBackend


class BedrockLLMBackend(LLMBackend):
    """AWS Bedrock LLM backend."""
    
    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        try:
            import boto3
            self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
            self.model_id = model_id
        except ImportError:
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response using AWS Bedrock."""
        system_prompt = """You are a cybersecurity analyst assistant. You have access to Shodan scan data that shows internet-connected devices, their services, versions, and locations.

Use the provided context data to answer questions about:
- Network services and their versions
- Geographic distribution of services
- Potential security vulnerabilities
- Infrastructure patterns
- Service configurations

Be specific and cite the data when possible. If the context doesn't contain enough information to answer the question, say so clearly."""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4000,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Context Data:\n{context}\n\nUser Question: {prompt}\n\nPlease provide a detailed analysis based on the scan data above."
                    }
                ]
            }
            
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            result = json.loads(response['body'].read())
            return result['content'][0]['text']
        except Exception as e:
            return f"Error generating response: {e}"