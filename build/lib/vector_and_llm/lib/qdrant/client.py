#!/usr/bin/env python3
"""
Qdrant client for vector operations.
"""

import json
import uuid
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from ..embedding.base import EmbeddingBackend
from ..llm.base import LLMBackend
from .utils import create_embedding_text, format_search_results


class QdrantClient:
    """Qdrant client for vector database operations."""
    
    def __init__(self, 
                 host: str = "127.0.0.1", 
                 port: int = 6333, 
                 collection_names: List[str] = None,
                 embedding_backend: Optional[EmbeddingBackend] = None):
        self.qdrant_url = f"http://{host}:{port}"
        self.collection_names = collection_names or ["shodan_query"]
        self.embedding_backend = embedding_backend
        self.vector_size = embedding_backend.get_vector_size() if embedding_backend else 384
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    @property
    def collection_name(self):
        """Backward compatibility - returns first collection name."""
        return self.collection_names[0]
    
    def collection_exists(self, collection_name: str = None) -> bool:
        """Check if the collection exists."""
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            print(f"🔍 Checking if collection '{collection_name}' exists...")
            response = self.session.get(f"{self.qdrant_url}/collections/{collection_name}")
            exists = response.status_code == 200
            print(f"🔍 Collection existence check: {exists} (status: {response.status_code})")
            return exists
        except requests.RequestException as e:
            print(f"✗ Error checking collection: {e}")
            return False
    
    def create_collection(self, collection_name: str = None) -> bool:
        """Create a new collection with the specified vector size."""
        if collection_name is None:
            collection_name = self.collection_name
            
        collection_config = {
            "vectors": {
                "size": self.vector_size,
                "distance": "Cosine"
            }
        }
        
        try:
            print(f"🔧 Attempting to create collection with config: {collection_config}")
            response = self.session.put(
                f"{self.qdrant_url}/collections/{collection_name}",
                json=collection_config
            )
            print(f"🔧 Collection creation response: {response.status_code} - {response.text}")
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"✗ Error creating collection: {e}")
            return False
    
    def ensure_collection_exists(self, collection_name: str = None) -> bool:
        """Ensure the collection exists, create it if it doesn't."""
        if collection_name is None:
            collection_name = self.collection_name
            
        if self.collection_exists(collection_name):
            print(f"✅ Collection '{collection_name}' already exists")
            return True
        
        print(f"🔧 Collection '{collection_name}' does not exist, creating...")
        if self.create_collection(collection_name):
            print(f"✅ Collection '{collection_name}' created successfully (vector size: {self.vector_size})")
            return True
        else:
            print(f"✗ Failed to create collection '{collection_name}'")
            return False
    
    def create_point(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Shodan match to a Qdrant point format."""
        # Generate embedding from match data
        embedding_text = create_embedding_text(match)
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
    
    def upload_point(self, point: Dict[str, Any], collection_name: str = None) -> Tuple[bool, str]:
        """Upload a single point to Qdrant."""
        if collection_name is None:
            collection_name = self.collection_name
            
        try:
            response = self.session.put(
                f"{self.qdrant_url}/collections/{collection_name}/points",
                json=point
            )
            if response.status_code == 200:
                return True, "Success"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as e:
            return False, f"Request error: {e}"
    
    def search_similar(self, query: str, top_k: int = 5, debug: bool = False, collection_names: List[str] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors across collections."""
        if collection_names is None:
            collection_names = self.collection_names
            
        if len(collection_names) == 1:
            return self._search_single_collection(query, top_k, debug, collection_names[0])
        else:
            return self._search_multi_collection(query, top_k, debug, collection_names)
    
    def _search_single_collection(self, query: str, top_k: int = 5, debug: bool = False, collection_name: str = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in a single Qdrant collection."""
        if collection_name is None:
            collection_name = self.collection_names[0]
            
        if debug:
            print(f"🔍 DEBUG: Generating embedding for query: '{query}'")
        else:
            print(f"🔍 Generating embedding for query: '{query}'")
        
        # Generate embedding for the query
        query_vector = self.embedding_backend.get_embedding(query)
        
        # Validate vector size
        if len(query_vector) != self.vector_size:
            print(f"⚠️  WARNING: Vector size mismatch!")
            print(f"   Expected: {self.vector_size} dims")
            print(f"   Got: {len(query_vector)} dims")
            print(f"   This may cause search to fail. Ensure you use the SAME embedding backend")
            print(f"   for both data import (shodan-to-qdrant) and search (llm-rag-shodan).")
        
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
            
            if response.status_code != 200:
                error_msg = response.text
                print(f"✗ Search failed with status {response.status_code}: {error_msg}")
                if "vector dimensionality" in error_msg.lower() or "dimension" in error_msg.lower():
                    print(f"🔴 CRITICAL: Vector dimension mismatch detected!")
                    print(f"   Collection expects {self.vector_size} dimensions")
                    print(f"   But query vector has {len(query_vector)} dimensions")
                return []
            
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
    
    def _search_multi_collection(self, query: str, top_k: int = 5, debug: bool = False, collection_names: List[str] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors across multiple collections."""
        if collection_names is None:
            collection_names = self.collection_names
            
        if debug:
            print(f"🔍 DEBUG: Generating embedding for query: '{query}'")
        else:
            print(f"🔍 Generating embedding for query: '{query}'")
        
        # Generate embedding for the query
        query_vector = self.embedding_backend.get_embedding(query)
        
        # Validate vector size
        if len(query_vector) != self.vector_size:
            print(f"⚠️  WARNING: Vector size mismatch!")
            print(f"   Expected: {self.vector_size} dims")
            print(f"   Got: {len(query_vector)} dims")
            print(f"   This may cause search to fail. Ensure you use the SAME embedding backend")
            print(f"   for both data import (shodan-to-qdrant) and search (llm-rag-shodan).")
        
        if debug:
            print(f"🔍 DEBUG: Query vector size: {len(query_vector)}")
            print(f"🔍 DEBUG: First 5 vector values: {query_vector[:5]}")
        
        all_results = []
        
        # Search each collection
        for collection_name in collection_names:
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
                    error_msg = response.text
                    print(f"⚠️  Collection '{collection_name}' returned status {response.status_code}")
                    if "vector dimensionality" in error_msg.lower() or "dimension" in error_msg.lower():
                        print(f"🔴 CRITICAL: Vector dimension mismatch in collection '{collection_name}'!")
                        print(f"   Collection expects {self.vector_size} dimensions")
                        print(f"   But query vector has {len(query_vector)} dimensions")
                    
            except requests.RequestException as e:
                print(f"⚠️  Error searching collection '{collection_name}': {e}")
                continue
        
        # Sort all results by similarity score (descending) and take top_k
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = all_results[:top_k]
        
        print(f"✅ Found {len(final_results)} total results across {len(collection_names)} collections")
        
        if debug and final_results:
            print(f"🔍 DEBUG: Top result similarity score: {final_results[0].get('score', 'N/A')}")
            print(f"🔍 DEBUG: Top result from collection: {final_results[0].get('collection', 'Unknown')}")
            if final_results[0].get('payload', {}).get('embedding_text'):
                print(f"🔍 DEBUG: Top result embedding text: {final_results[0]['payload']['embedding_text'][:200]}...")
        
        return final_results
    
    def process_file(self, file_path: Path, collection_name: str = None) -> Tuple[int, int]:
        """Process a single Shodan JSON file."""
        if collection_name is None:
            collection_name = self.collection_name
            
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
            if self.embedding_backend.__class__.__name__ in ['OllamaEmbeddingBackend', 'BedrockEmbeddingBackend']:
                time.sleep(0.1)
            
            point = self.create_point(match)
            
            success, message = self.upload_point(point, collection_name)
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
    
    def process_files(self, file_paths: List[Path], collection_name: str = None) -> None:
        """Process multiple Shodan JSON files."""
        if collection_name is None:
            collection_name = self.collection_name
            
        print(f"🚀 Starting import of {len(file_paths)} file(s) to Qdrant collection '{collection_name}'")
        print(f"🔗 Qdrant URL: {self.qdrant_url}")
        
        # Ensure collection exists before processing
        print("🔍 Checking collection status...")
        if not self.ensure_collection_exists(collection_name):
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
            success, errors = self.process_file(file_path, collection_name)
            total_success += success
            total_errors += errors
        
        print(f"\n🎉 Import complete!")
        print(f"📊 Total results: {total_success} successful uploads, {total_errors} errors")
        print(f"📁 Files processed: {total_files}")