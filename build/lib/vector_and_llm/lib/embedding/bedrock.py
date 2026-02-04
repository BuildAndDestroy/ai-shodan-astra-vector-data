#!/usr/bin/env python3
"""
AWS Bedrock embedding backend using Titan Embeddings.
"""

import json
from typing import List
from .base import EmbeddingBackend


class BedrockEmbeddingBackend(EmbeddingBackend):
    """AWS Bedrock embedding backend using Titan Embeddings."""
    
    def __init__(self, region_name: str = "us-east-1", model_id: str = "amazon.titan-embed-text-v1"):
        try:
            import boto3
            self.bedrock = boto3.client('bedrock-runtime', region_name=region_name)
            self.model_id = model_id
            self.vector_size = 1536  # Titan Embeddings dimensions
        except ImportError:
            raise ImportError("boto3 is required for AWS Bedrock. Install with: pip install boto3")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from AWS Bedrock."""
        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    'inputText': text
                })
            )
            result = json.loads(response['body'].read())
            return result['embedding']
        except Exception as e:
            print(f"✗ Bedrock embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * self.vector_size