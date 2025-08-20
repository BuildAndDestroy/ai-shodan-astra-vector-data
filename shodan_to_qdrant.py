#!/usr/bin/env python3
"""
Shodan to Qdrant Importer with Configurable Embeddings
Imports multiple Shodan JSON files into a Qdrant vector database collection.
Supports AWS Bedrock (Titan) and Ollama (nomic-embed-text) embeddings.

Usage:
    python shodan_importer.py file1.json file2.json file3.json
    python shodan_importer.py --embedding-backend ollama *.json
    python shodan_importer.py --embedding-backend bedrock --aws-region us-east-1 *.json
"""

import json
import uuid
import requests
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time


class EmbeddingBackend:
    """Base class for embedding backends."""
    
    def __init__(self):
        self.vector_size = 384  # Default, overridden by implementations
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def get_vector_size(self) -> int:
        """Get the vector size for this embedding model."""
        return self.vector_size


class ZeroEmbeddingBackend(EmbeddingBackend):
    """Zero vector backend (no real embeddings)."""
    
    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size
    
    def get_embedding(self, text: str) -> List[float]:
        return [0.0] * self.vector_size


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embedding backend using nomic-embed-text."""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "nomic-embed-text"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self.vector_size = 768  # nomic-embed-text dimensions
        
    def get_embedding(self, text: str) -> List[float]:
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            print(f"✗ Ollama embedding error: {e}")
            # Return zero vector as fallback
            return [0.0] * self.vector_size


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


class ShodanQdrantImporter:
    def __init__(self, 
                 qdrant_host: str = "10.0.20.62", 
                 qdrant_port: int = 6333, 
                 collection_name: str = "shodan_query",
                 embedding_backend: Optional[EmbeddingBackend] = None):
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self.collection_name = collection_name
        self.embedding_backend = embedding_backend or ZeroEmbeddingBackend()
        self.vector_size = self.embedding_backend.get_vector_size()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        print(f"🔧 Using embedding backend: {self.embedding_backend.__class__.__name__}")
        print(f"🔧 Vector size: {self.vector_size}")
        
    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        try:
            print(f"🔍 Checking if collection '{self.collection_name}' exists...")
            response = self.session.get(f"{self.qdrant_url}/collections/{self.collection_name}")
            exists = response.status_code == 200
            print(f"🔍 Collection existence check: {exists} (status: {response.status_code})")
            return exists
        except requests.RequestException as e:
            print(f"✗ Error checking collection: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create a new collection with the specified vector size."""
        collection_config = {
            "vectors": {
                "size": self.vector_size,
                "distance": "Cosine"
            }
        }
        
        try:
            print(f"🔧 Attempting to create collection with config: {collection_config}")
            response = self.session.put(
                f"{self.qdrant_url}/collections/{self.collection_name}",
                json=collection_config
            )
            print(f"🔧 Collection creation response: {response.status_code} - {response.text}")
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"✗ Error creating collection: {e}")
            return False
    
    def ensure_collection_exists(self) -> bool:
        """Ensure the collection exists, create it if it doesn't."""
        if self.collection_exists():
            print(f"✅ Collection '{self.collection_name}' already exists")
            return True
        
        print(f"🔧 Collection '{self.collection_name}' does not exist, creating...")
        if self.create_collection():
            print(f"✅ Collection '{self.collection_name}' created successfully (vector size: {self.vector_size})")
            return True
        else:
            print(f"✗ Failed to create collection '{self.collection_name}'")
            return False
    
    def create_embedding_text(self, match: Dict[str, Any]) -> str:
        """Create text string from Shodan match for embedding."""
        # Combine key fields for semantic embedding
        parts = []
        
        if match.get("ip_str"):
            parts.append(f"IP: {match['ip_str']}")
        
        if match.get("port"):
            parts.append(f"Port: {match['port']}")
            
        if match.get("product"):
            parts.append(f"Service: {match['product']}")
            
        if match.get("version"):
            parts.append(f"Version: {match['version']}")
            
        if match.get("location"):
            location = match["location"]
            if location.get("country_name"):
                parts.append(f"Country: {location['country_name']}")
            if location.get("city"):
                parts.append(f"City: {location['city']}")
        
        # Add SSH-specific info if available
        if match.get("ssh") and match["ssh"].get("type"):
            parts.append(f"SSH Key Type: {match['ssh']['type']}")
            
        # Optionally include banner data (truncated to avoid token limits)
        if match.get("data"):
            banner = match["data"][:500]  # First 500 chars
            parts.append(f"Banner: {banner}")
        
        return " | ".join(parts)
    
    def create_point(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Shodan match to a Qdrant point format."""
        # Generate embedding from match data
        embedding_text = self.create_embedding_text(match)
        vector = self.embedding_backend.get_embedding(embedding_text)
        
        return {
            "points": [{
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {
                    "ip_str": match.get("ip_str"),
                    "port": match.get("port"),
                    "product": match.get("product"),
                    "version": match.get("version"),
                    "timestamp": match.get("timestamp"),
                    "location": match.get("location"),
                    "ssh": match.get("ssh"),
                    "unique_key": f"{match.get('ip_str', '')}_{match.get('port', '')}",
                    "embedding_text": embedding_text  # Store the text used for embedding
                }
            }]
        }
    
    def upload_point(self, point: Dict[str, Any]) -> Tuple[bool, str]:
        """Upload a single point to Qdrant."""
        try:
            response = self.session.put(
                f"{self.qdrant_url}/collections/{self.collection_name}/points",
                json=point
            )
            if response.status_code == 200:
                return True, "Success"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            return False, f"Request error: {e}"
    
    def process_file(self, file_path: Path) -> Tuple[int, int]:
        """Process a single Shodan JSON file."""
        print(f"\n📁 Processing file: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"✗ Error reading {file_path}: {e}")
            return 0, 0
        
        matches = data.get("matches", [])
        if not matches:
            print(f"⚠️  No matches found in {file_path}")
            return 0, 0
        
        print(f"📊 Found {len(matches)} records to process")
        
        success_count = 0
        error_count = 0
        
        for i, match in enumerate(matches, 1):
            # Add small delay for API rate limiting
            if isinstance(self.embedding_backend, (OllamaEmbeddingBackend, BedrockEmbeddingBackend)):
                time.sleep(0.1)
            
            point = self.create_point(match)
            
            success, message = self.upload_point(point)
            if success:
                success_count += 1
                ip = match.get('ip_str', 'unknown')
                print(f"✓ Record {i}/{len(matches)} ({ip}): Success")
            else:
                error_count += 1
                print(f"✗ Record {i}/{len(matches)}: {message}")
            
            # Progress indicator every 10 records
            if i % 10 == 0:
                print(f"--- Progress: {i}/{len(matches)} records processed ---")
        
        print(f"📈 File {file_path.name} complete: {success_count} success, {error_count} errors")
        return success_count, error_count
    
    def process_files(self, file_paths: List[Path]) -> None:
        """Process multiple Shodan JSON files."""
        print(f"🚀 Starting import of {len(file_paths)} file(s) to Qdrant collection '{self.collection_name}'")
        print(f"🔗 Qdrant URL: {self.qdrant_url}")
        
        # Ensure collection exists before processing
        print("🔍 Checking collection status...")
        if not self.ensure_collection_exists():
            print("❌ Cannot proceed without a valid collection")
            return
        
        print("✅ Collection ready, starting file processing...")
        
        total_success = 0
        total_errors = 0
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths, 1):
            if not file_path.exists():
                print(f"⚠️  File not found: {file_path}")
                continue
            
            if not file_path.suffix.lower() == '.json':
                print(f"⚠️  Skipping non-JSON file: {file_path}")
                continue
            
            print(f"\n📂 Processing file {i}/{total_files}")
            success, errors = self.process_file(file_path)
            total_success += success
            total_errors += errors
        
        print(f"\n🎉 Import complete!")
        print(f"📊 Total results: {total_success} successful uploads, {total_errors} errors")
        print(f"📁 Files processed: {total_files}")


def main():
    parser = argparse.ArgumentParser(
        description="Import Shodan JSON files into Qdrant vector database with configurable embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Zero vectors (no embeddings)
  python shodan_importer.py *.json
  
  # Ollama embeddings
  python shodan_importer.py --embedding-backend ollama --ollama-host localhost *.json
  
  # AWS Bedrock embeddings
  python shodan_importer.py --embedding-backend bedrock --aws-region us-east-1 *.json
  
  # Custom vector size with zero vectors
  python shodan_importer.py --embedding-backend zero --vector-size 1536 *.json
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more Shodan JSON files to import"
    )
    
    parser.add_argument(
        "--host",
        default="10.0.20.62",
        help="Qdrant host (default: 10.0.20.62)"
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
        "--embedding-backend",
        choices=["zero", "ollama", "bedrock"],
        default="zero",
        help="Embedding backend to use (default: zero)"
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
        "--ollama-model",
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
        "--aws-model",
        default="amazon.titan-embed-text-v1",
        help="AWS Bedrock model ID (default: amazon.titan-embed-text-v1)"
    )
    
    args = parser.parse_args()
    
    # Create embedding backend based on choice
    if args.embedding_backend == "zero":
        vector_size = args.vector_size or 384
        embedding_backend = ZeroEmbeddingBackend(vector_size)
    elif args.embedding_backend == "ollama":
        embedding_backend = OllamaEmbeddingBackend(
            host=args.ollama_host,
            port=args.ollama_port,
            model=args.ollama_model
        )
    elif args.embedding_backend == "bedrock":
        embedding_backend = BedrockEmbeddingBackend(
            region_name=args.aws_region,
            model_id=args.aws_model
        )
    
    # Debug: print configuration
    print(f"🔧 Configuration:")
    print(f"   Qdrant: {args.host}:{args.port}")
    print(f"   Collection: {args.collection}")
    print(f"   Embedding Backend: {args.embedding_backend}")
    print(f"   Files: {len(args.files)} file(s)")
    
    # Convert string paths to Path objects
    file_paths = [Path(f) for f in args.files]
    
    # Create importer and process files
    importer = ShodanQdrantImporter(
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_name=args.collection,
        embedding_backend=embedding_backend
    )
    
    importer.process_files(file_paths)


if __name__ == "__main__":
    main()
