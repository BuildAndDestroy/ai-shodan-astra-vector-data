#!/usr/bin/env python3
"""
Tool for importing Shodan JSON files into Qdrant vector database.

Usage:
    shodan-to-qdrant [options] file1.json file2.json ...
    python -m vector_and_llm.tools.shodan_to_qdrant [options] file1.json file2.json ...
"""

import argparse
from pathlib import Path
from typing import List

from ..lib.embedding import ZeroEmbeddingBackend, OllamaEmbeddingBackend, BedrockEmbeddingBackend
from ..lib.qdrant import QdrantClient


def create_embedding_backend(args):
    """Create embedding backend based on arguments."""
    if args.embedding_backend == "zero":
        vector_size = args.vector_size or 384
        return ZeroEmbeddingBackend(vector_size)
    elif args.embedding_backend == "ollama":
        return OllamaEmbeddingBackend(
            host=args.ollama_host,
            port=args.ollama_port,
            model=args.ollama_embed_model
        )
    elif args.embedding_backend == "bedrock":
        return BedrockEmbeddingBackend(
            region_name=args.aws_region,
            model_id=args.aws_embed_model
        )


def main():
    parser = argparse.ArgumentParser(
        description="Import Shodan JSON files into Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import with Ollama embeddings (multiple ways)
  shodan-to-qdrant --embedding-backend ollama *.json
  shodan-to-qdrant --embedding-backend ollama -f file1.json -f file2.json
  shodan-to-qdrant --embedding-backend ollama --file scan1.json --file scan2.json
  
  # Import with AWS Bedrock embeddings
  shodan-to-qdrant --embedding-backend bedrock --aws-region us-east-1 -f *.json
  
  # Import with zero vectors (testing)
  shodan-to-qdrant --embedding-backend zero --vector-size 768 -f scan.json
        """
    )
    
    # File arguments
    parser.add_argument(
        "files",
        nargs="*",
        help="Shodan JSON files to import"
    )
    
    parser.add_argument(
        "-f", "--file",
        action="append",
        help="Shodan JSON file to import (can be used multiple times)"
    )
    
    # Qdrant options
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Qdrant host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )
    
    parser.add_argument(
        "--collection",
        default="shodan_query",
        help="Qdrant collection name (default: shodan_query)"
    )
    
    # Embedding options
    parser.add_argument(
        "--embedding-backend",
        choices=["zero", "ollama", "bedrock"],
        default="ollama",
        help="Embedding backend to use (default: ollama)"
    )
    
    parser.add_argument(
        "--vector-size",
        type=int,
        help="Vector size (only used with 'zero' backend, others auto-detect)"
    )
    
    # Ollama-specific options
    parser.add_argument(
        "--ollama-host",
        default="localhost",
        help="Ollama host (default: localhost)"
    )
    
    parser.add_argument(
        "--ollama-port",
        type=int,
        default=11434,
        help="Ollama port (default: 11434)"
    )
    
    parser.add_argument(
        "--ollama-embed-model",
        default="nomic-embed-text",
        help="Ollama embedding model (default: nomic-embed-text)"
    )
    
    # AWS-specific options
    parser.add_argument(
        "--aws-region",
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)"
    )
    
    parser.add_argument(
        "--aws-embed-model",
        default="amazon.titan-embed-text-v1",
        help="AWS Bedrock embedding model ID (default: amazon.titan-embed-text-v1)"
    )
    
    args = parser.parse_args()
    
    # Combine file sources
    all_files = []
    if args.files:
        all_files.extend(args.files)
    if args.file:
        all_files.extend(args.file)
    
    # Validate arguments
    if not all_files:
        print("❌ Error: No files provided for import")
        parser.print_help()
        return
    
    # Create embedding backend
    embedding_backend = create_embedding_backend(args)
    
    # Create Qdrant client
    qdrant_client = QdrantClient(
        host=args.host,
        port=args.port,
        collection_names=[args.collection],
        embedding_backend=embedding_backend
    )
    
    print(f"🔧 Using embedding backend: {embedding_backend.__class__.__name__}")
    print(f"🔧 Vector size: {embedding_backend.get_vector_size()}")
    print(f"🔧 Collection: {args.collection}")
    
    # Process files
    file_paths = [Path(f) for f in all_files]
    qdrant_client.process_files(file_paths, args.collection)


if __name__ == "__main__":
    main()