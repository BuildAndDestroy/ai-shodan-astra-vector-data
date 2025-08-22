#!/usr/bin/env python3
"""
Tool for performing RAG queries on Shodan data stored in Qdrant.

Usage:
    llm-rag-shodan -p "What SSH servers are running in the US?" [options]
    python -m vector_and_llm.tools.llm_rag_shodan -p "Show me vulnerable Apache servers" [options]
"""

import argparse

from ..lib.embedding import ZeroEmbeddingBackend, OllamaEmbeddingBackend, BedrockEmbeddingBackend
from ..lib.llm import OllamaLLMBackend, BedrockLLMBackend
from ..lib.qdrant import QdrantClient, format_search_results


class ShodanRAG:
    """RAG system for querying Shodan data."""
    
    def __init__(self, qdrant_client, llm_backend):
        self.qdrant_client = qdrant_client
        self.llm_backend = llm_backend
    
    def query(self, prompt: str, top_k: int = 5, debug: bool = False) -> str:
        """Perform RAG query: retrieve similar data and generate response."""
        print(f"🤖 Processing RAG query: '{prompt}'")
        
        # Step 1: Retrieve similar data
        similar_results = self.qdrant_client.search_similar(prompt, top_k, debug)
        
        if not similar_results:
            return "I couldn't find any relevant data in the Shodan database to answer your question. The database might be empty or your query might not match any stored data."
        
        # Step 2: Format context
        context = format_search_results(similar_results)
        if debug:
            print(f"📄 DEBUG: Generated context:")
            print("-" * 40)
            print(context)
            print("-" * 40)
        else:
            print(f"📄 Generated context with {len(similar_results)} results")
        
        # Step 3: Generate response
        print(f"💭 Generating response using LLM...")
        response = self.llm_backend.generate_response(prompt, context)
        
        print(f"✅ RAG query complete")
        return response


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


def create_llm_backend(args):
    """Create LLM backend based on arguments."""
    if args.llm_backend == "ollama":
        return OllamaLLMBackend(
            host=args.ollama_host,
            port=args.ollama_port,
            model=args.ollama_llm_model
        )
    elif args.llm_backend == "bedrock":
        return BedrockLLMBackend(
            region_name=args.aws_region,
            model_id=args.aws_llm_model
        )


def main():
    parser = argparse.ArgumentParser(
        description="Perform RAG queries on Shodan data in Qdrant vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic query
  llm-rag-shodan -p "What SSH servers are running in the US?" --collection my_collection
  
  # Query multiple collections
  llm-rag-shodan -p "Compare SSH servers across datasets" --collections collection1 collection2 collection3
  
  # Query with AWS Bedrock LLM
  llm-rag-shodan -p "Show me vulnerable Apache servers" --llm-backend bedrock --collections scan1 scan2
  
  # Debug mode
  llm-rag-shodan -p "Show me all services in Moscow" --debug --top-k 50
        """
    )
    
    # Core arguments
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Query prompt for RAG"
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
    
    parser.add_argument(
        "--collections",
        nargs="+",
        help="Multiple Qdrant collection names to search (overrides --collection)"
    )
    
    # Query options
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar results to retrieve for RAG (default: 5)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing RAG pipeline details"
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
    
    # LLM options
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "bedrock"],
        default="ollama",
        help="LLM backend for RAG queries (default: ollama)"
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
    
    parser.add_argument(
        "--ollama-llm-model",
        default="llama3.2",
        help="Ollama LLM model (default: llama3.2)"
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
    
    parser.add_argument(
        "--aws-llm-model",
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        help="AWS Bedrock LLM model ID (default: anthropic.claude-3-sonnet-20240229-v1:0)"
    )
    
    args = parser.parse_args()
    
    # Determine collections to use
    if args.collections:
        collection_names = args.collections
    else:
        collection_names = [args.collection]
    
    # Create backends
    embedding_backend = create_embedding_backend(args)
    llm_backend = create_llm_backend(args)
    
    # Create Qdrant client
    qdrant_client = QdrantClient(
        host=args.host,
        port=args.port,
        collection_names=collection_names,
        embedding_backend=embedding_backend
    )
    
    print(f"🔧 Using embedding backend: {embedding_backend.__class__.__name__}")
    print(f"🔧 Using LLM backend: {llm_backend.__class__.__name__}")
    print(f"🔧 Vector size: {embedding_backend.get_vector_size()}")
    print(f"🔧 Collections: {', '.join(collection_names)}")
    
    # Create RAG system
    rag_system = ShodanRAG(qdrant_client, llm_backend)
    
    print(f"🔍 RAG Query Mode")
    print(f"📄 Query: {args.prompt}")
    print(f"🔢 Top-K: {args.top_k}")
    print(f"🗂️  Collections: {', '.join(collection_names)}")
    print("-" * 50)
    
    # Execute query
    response = rag_system.query(args.prompt, args.top_k, args.debug)
    print(f"\n💬 Response:")
    print(response)


if __name__ == "__main__":
    main()