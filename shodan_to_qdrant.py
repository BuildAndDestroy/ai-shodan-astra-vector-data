#!/usr/bin/env python3
"""
Shodan to Qdrant Importer with RAG Query Support
Imports multiple Shodan JSON files into a Qdrant vector database collection
and provides RAG-based querying capabilities.

Usage:
    # Import data
    python shodan_rag.py file1.json file2.json file3.json
    
    # Query with RAG
    python shodan_rag.py -p "What SSH servers are running in the US?"
    python shodan_rag.py -p "Show me vulnerable Apache servers" --llm-backend bedrock
    python shodan_rag.py -p "What's the security posture in New York?" --top-k 10
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


class LLMBackend:
    """Base class for LLM backends."""
    
    def generate_response(self, prompt: str, context: str) -> str:
        """Generate response given prompt and context. Must be implemented by subclasses."""
        raise NotImplementedError


class OllamaLLMBackend(LLMBackend):
    """Ollama LLM backend."""
    
    def __init__(self, host: str = "localhost", port: int = 11434, model: str = "llama3.2"):
        self.base_url = f"http://{host}:{port}"
        self.model = model
    
    def generate_response(self, prompt: str, context: str) -> str:
        system_prompt = """You are a cybersecurity analyst assistant. You have access to Shodan scan data that shows internet-connected devices, their services, versions, and locations.

Use the provided context data to answer questions about:
- Network services and their versions
- Geographic distribution of services
- Potential security vulnerabilities
- Infrastructure patterns
- Service configurations

Be specific and cite the data when possible. If the context doesn't contain enough information to answer the question, say so clearly."""

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


class ShodanQdrantRAG:
    def __init__(self, 
                 qdrant_host: str = "10.0.20.62", 
                 qdrant_port: int = 6333, 
                 collection_names: List[str] = None,
                 embedding_backend: Optional[EmbeddingBackend] = None,
                 llm_backend: Optional[LLMBackend] = None):
        self.qdrant_url = f"http://{qdrant_host}:{qdrant_port}"
        self.collection_names = collection_names or ["shodan_query"]
        self.embedding_backend = embedding_backend or ZeroEmbeddingBackend()
        self.llm_backend = llm_backend or OllamaLLMBackend()
        self.vector_size = self.embedding_backend.get_vector_size()
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        print(f"🔧 Using embedding backend: {self.embedding_backend.__class__.__name__}")
        print(f"🔧 Using LLM backend: {self.llm_backend.__class__.__name__}")
        print(f"🔧 Vector size: {self.vector_size}")
        print(f"🔧 Collections: {', '.join(self.collection_names)}")
    
    @property
    def collection_name(self):
        """Backward compatibility - returns first collection name."""
        return self.collection_names[0]
    
    def search_similar_multi_collection(self, query: str, top_k: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
        """Search for similar vectors across multiple collections in Qdrant based on query."""
        if debug:
            print(f"🔍 DEBUG: Generating embedding for query: '{query}'")
        else:
            print(f"🔍 Generating embedding for query: '{query}'")
        
        # Generate embedding for the query
        query_vector = self.embedding_backend.get_embedding(query)
        
        if debug:
            print(f"🔍 DEBUG: Query vector size: {len(query_vector)}")
            print(f"🔍 DEBUG: First 5 vector values: {query_vector[:5]}")
        
        all_results = []
        
        # Search each collection
        for collection_name in self.collection_names:
            if debug:
                print(f"🔍 DEBUG: Searching collection '{collection_name}'")
            
            search_payload = {
                "vector": query_vector,
                "limit": top_k,
                "with_payload": True
            }
            
            try:
                response = self.session.post(
                    f"{self.qdrant_url}/collections/{collection_name}/points/search",
                    json=search_payload
                )
                
                if response.status_code == 200:
                    collection_results = response.json().get("result", [])
                    # Add collection name to each result for context
                    for result in collection_results:
                        result['collection'] = collection_name
                    all_results.extend(collection_results)
                    
                    if debug:
                        print(f"🔍 DEBUG: Found {len(collection_results)} results in '{collection_name}'")
                else:
                    print(f"⚠️  Collection '{collection_name}' returned status {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"⚠️  Error searching collection '{collection_name}': {e}")
                continue
        
        # Sort all results by similarity score (descending) and take top_k
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = all_results[:top_k]
        
        print(f"✅ Found {len(final_results)} total results across {len(self.collection_names)} collections")
        
        if debug and final_results:
            print(f"🔍 DEBUG: Top result similarity score: {final_results[0].get('score', 'N/A')}")
            print(f"🔍 DEBUG: Top result from collection: {final_results[0].get('collection', 'Unknown')}")
            if final_results[0].get('payload', {}).get('embedding_text'):
                print(f"🔍 DEBUG: Top result embedding text: {final_results[0]['payload']['embedding_text'][:200]}...")
        
        return final_results
    
    def search_similar(self, query: str, top_k: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
        """Search for similar vectors in Qdrant based on query (backward compatibility)."""
        if len(self.collection_names) == 1:
            # Single collection - use original method
            return self.search_similar_single_collection(query, top_k, debug, self.collection_names[0])
        else:
            # Multiple collections - use multi-collection method
            return self.search_similar_multi_collection(query, top_k, debug)
    
    def search_similar_single_collection(self, query: str, top_k: int = 5, debug: bool = False, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in a single Qdrant collection based on query."""
        if collection_name is None:
            collection_name = self.collection_names[0]
            
        if debug:
            print(f"🔍 DEBUG: Generating embedding for query: '{query}'")
        else:
            print(f"🔍 Generating embedding for query: '{query}'")
        
        # Generate embedding for the query
        query_vector = self.embedding_backend.get_embedding(query)
        
        if debug:
            print(f"🔍 DEBUG: Query vector size: {len(query_vector)}")
            print(f"🔍 DEBUG: First 5 vector values: {query_vector[:5]}")
        
        # Search Qdrant
        search_payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True
        }
        
        try:
            print(f"🔍 Searching Qdrant collection '{collection_name}' for {top_k} most similar results...")
            response = self.session.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/search",
                json=search_payload
            )
            response.raise_for_status()
            
            results = response.json().get("result", [])
            # Add collection name for consistency
            for result in results:
                result['collection'] = collection_name
                
            print(f"✅ Found {len(results)} similar results")
            
            if debug and results:
                print(f"🔍 DEBUG: Top result similarity score: {results[0].get('score', 'N/A')}")
                print(f"🔍 DEBUG: Top result payload keys: {list(results[0].get('payload', {}).keys())}")
                if results[0].get('payload', {}).get('embedding_text'):
                    print(f"🔍 DEBUG: Top result embedding text: {results[0]['payload']['embedding_text'][:200]}...")
            
            return results
            
        except requests.RequestException as e:
            print(f"✗ Error searching Qdrant: {e}")
            return []
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into readable context for LLM."""
        if not results:
            return "No relevant data found in the database."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            payload = result.get("payload", {})
            score = result.get("score", 0)
            collection = result.get("collection", "unknown")
            
            # Format individual result
            parts = []
            if payload.get("ip_str"):
                parts.append(f"IP: {payload['ip_str']}")
            if payload.get("port"):
                parts.append(f"Port: {payload['port']}")
            if payload.get("product"):
                parts.append(f"Service: {payload['product']}")
            if payload.get("version"):
                parts.append(f"Version: {payload['version']}")
            
            location = payload.get("location", {})
            if location:
                loc_parts = []
                if location.get("city"):
                    loc_parts.append(location["city"])
                if location.get("country_name"):
                    loc_parts.append(location["country_name"])
                if loc_parts:
                    parts.append(f"Location: {', '.join(loc_parts)}")
            
            if payload.get("timestamp"):
                parts.append(f"Timestamp: {payload['timestamp']}")
            
            # Add collection information
            parts.append(f"Dataset: {collection}")
            
            result_text = f"Result {i} (similarity: {score:.3f}): {' | '.join(parts)}"
            context_parts.append(result_text)
        
        return "\n".join(context_parts)
    
    def query(self, prompt: str, top_k: int = 5, debug: bool = False) -> str:
        """Perform RAG query: retrieve similar data and generate response."""
        print(f"🤖 Processing RAG query: '{prompt}'")
        
        # Step 1: Retrieve similar data
        similar_results = self.search_similar(prompt, top_k, debug)
        
        if not similar_results:
            return "I couldn't find any relevant data in the Shodan database to answer your question. The database might be empty or your query might not match any stored data."
        
        # Step 2: Format context
        context = self.format_search_results(similar_results)
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
    
    # Import functionality (same as before)
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
            
        # Enhanced location handling with multiple variations
        if match.get("location"):
            location = match["location"]
            if location.get("country_name"):
                country = location["country_name"]
                parts.append(f"Country: {country}")
                
                # Add common country name variations for better matching
                country_variations = {
                    "Russian Federation": ["Russia", "Russian Federation", "RU"],
                    "United States": ["USA", "US", "United States", "America"],
                    "United Kingdom": ["UK", "Britain", "United Kingdom", "GB"],
                    "China": ["China", "People's Republic of China", "PRC", "CN"],
                    "Germany": ["Germany", "Deutschland", "DE"],
                    # Add more as needed
                }
                
                for full_name, variations in country_variations.items():
                    if country == full_name:
                        parts.extend([f"Location: {var}" for var in variations])
                        break
                
            if location.get("country_code"):
                parts.append(f"Country Code: {location['country_code']}")
                
            if location.get("city"):
                parts.append(f"City: {location['city']}")
                
            # Add geographic context
            if location.get("country_name") and location.get("city"):
                parts.append(f"Geographic Location: {location['city']}, {location['country_name']}")
        
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
            print("⌛ Cannot proceed without a valid collection")
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
        description="Import Shodan JSON files into Qdrant vector database and query with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import data with Ollama embeddings (multiple ways)
  python shodan_rag.py --embedding-backend ollama *.json
  python shodan_rag.py --embedding-backend ollama -f file1.json -f file2.json
  python shodan_rag.py --embedding-backend ollama --file scan1.json --file scan2.json
  
  # Query single collection with RAG
  python shodan_rag.py -p "What SSH servers are running in the US?" --collection my_collection
  
  # Query multiple collections with RAG
  python shodan_rag.py -p "Compare SSH servers across datasets" --collections collection1 collection2 collection3
  
  # Query with AWS Bedrock LLM
  python shodan_rag.py -p "Show me vulnerable Apache servers" --llm-backend bedrock --collections scan1 scan2
  
  # Import with Bedrock embeddings, query with Bedrock LLM across multiple collections
  python shodan_rag.py --embedding-backend bedrock --llm-backend bedrock -f *.json
        """
    )
    
    # Core arguments
    parser.add_argument(
        "files",
        nargs="*",
        help="Shodan JSON files to import (omit when using -p for queries)"
    )
    
    parser.add_argument(
        "-f", "--file",
        action="append",
        help="Shodan JSON file to import (can be used multiple times)"
    )
    
    parser.add_argument(
        "-p", "--prompt",
        help="Query prompt for RAG (instead of importing files)"
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
        "--collections",
        nargs="+",
        help="Multiple Qdrant collection names to search (overrides --collection)"
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
    
    # Combine file sources
    all_files = []
    if args.files:
        all_files.extend(args.files)
    if args.file:
        all_files.extend(args.file)
    
    # Validate arguments
    if not args.prompt and not all_files:
        print("❌ Error: Either provide files to import OR use -p for queries")
        parser.print_help()
        return
    
    if args.prompt and all_files:
        print("❌ Error: Cannot import files and query at the same time")
        parser.print_help()
        return
    
    # Create embedding backend
    if args.embedding_backend == "zero":
        vector_size = args.vector_size or 384
        embedding_backend = ZeroEmbeddingBackend(vector_size)
    elif args.embedding_backend == "ollama":
        embedding_backend = OllamaEmbeddingBackend(
            host=args.ollama_host,
            port=args.ollama_port,
            model=args.ollama_embed_model
        )
    elif args.embedding_backend == "bedrock":
        embedding_backend = BedrockEmbeddingBackend(
            region_name=args.aws_region,
            model_id=args.aws_embed_model
        )
    
    # Create LLM backend for queries
    llm_backend = None
    if args.prompt:
        if args.llm_backend == "ollama":
            llm_backend = OllamaLLMBackend(
                host=args.ollama_host,
                port=args.ollama_port,
                model=args.ollama_llm_model
            )
        elif args.llm_backend == "bedrock":
            llm_backend = BedrockLLMBackend(
                region_name=args.aws_region,
                model_id=args.aws_llm_model
            )
    
    # Determine collections to use
    if args.collections:
        collection_names = args.collections
    else:
        collection_names = [args.collection]
    
    # Create RAG system
    rag_system = ShodanQdrantRAG(
        qdrant_host=args.host,
        qdrant_port=args.port,
        collection_names=collection_names,
        embedding_backend=embedding_backend,
        llm_backend=llm_backend
    )
    
    # Handle query or import
    if args.prompt:
        print(f"🔍 RAG Query Mode")
        print(f"📄 Query: {args.prompt}")
        print(f"🔢 Top-K: {args.top_k}")
        print(f"🗂️  Collections: {', '.join(collection_names)}")
        print("-" * 50)
        
        response = rag_system.query(args.prompt, args.top_k, args.debug)
        print(f"\n💬 Response:")
        print(response)
        
    else:
        print(f"📥 Import Mode")
        file_paths = [Path(f) for f in all_files]
        rag_system.process_files(file_paths)


if __name__ == "__main__":
    main()